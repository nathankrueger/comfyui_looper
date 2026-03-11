import io
import json as json_mod
import os
import re
import shutil
import threading
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename
from interactive.loop_state import LoopState, LoopStatus
from interactive.app_state import AppState
from image_processing.animator import make_animation
from utils.json_spec import (
    EMPTY_OBJECT, EMPTY_LIST, SettingsManager,
    serialize_override_value, deserialize_override_value,
)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def _strip_json_comments(text: str) -> str:
    """Remove single-line // comments from JSON text (same as SettingsManager)."""
    return re.sub(r'^\s*//.*$', '', text, flags=re.MULTILINE)


def _loop_img_filename(idx: int) -> str:
    return f"loop_img_{idx:06}.png"


def _get_data_dir() -> str:
    """Return the path to the data/ directory at project root."""
    # interactive/flask_app.py -> interactive/ -> comfyui_looper/ -> project root
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')


def create_app(app_state: AppState) -> Flask:
    app = Flask(__name__, static_folder='static', static_url_path='/static')

    def _require_loop_state():
        """Return the active LoopState or None. Caller should check and return 503."""
        return app_state.get_loop_state()

    # --- Routing ---

    @app.route('/')
    def index():
        if app_state.is_loop_running():
            return send_from_directory(app.static_folder, 'index.html')
        else:
            return send_from_directory(app.static_folder, 'picker.html')

    # --- App-level status ---

    @app.route('/api/app-status')
    def api_app_status():
        return jsonify({
            'loop_running': app_state.is_loop_running(),
            'json_file': app_state.get_current_json_name(),
            'output_folder': app_state.get_current_output_folder(),
        })

    # --- Workflow management ---

    @app.route('/api/workflows')
    def api_workflows():
        data_dir = _get_data_dir()
        user_dir = os.path.join(data_dir, 'user')
        workflows = []

        def _scan_dir(directory, source_label):
            if not os.path.isdir(directory):
                return
            for f in sorted(os.listdir(directory)):
                if not f.endswith('.json'):
                    continue
                full_path = os.path.join(directory, f)
                if not os.path.isfile(full_path):
                    continue
                try:
                    sm = SettingsManager(full_path, {})
                    sm.validate()
                    total = sm.get_total_iterations()
                except Exception:
                    total = None
                workflows.append({
                    'name': f,
                    'path': full_path,
                    'total_iterations': total,
                    'source': source_label,
                })

        _scan_dir(data_dir, 'built-in')
        _scan_dir(user_dir, 'user')
        return jsonify({'workflows': workflows})

    @app.route('/api/start', methods=['POST'])
    def api_start():
        if app_state.is_loop_running():
            return jsonify({'error': 'Loop already running'}), 409
        data = request.get_json()
        json_file = data.get('json_file')
        if not json_file:
            return jsonify({'error': 'json_file required'}), 400
        output_folder = data.get('output_folder') or None
        input_img = data.get('input_img')  # None=CLI default, ""=txt2img, path=use image
        # Validate image path if provided and non-empty
        if input_img:
            img_dir = os.path.abspath(os.path.join(_get_data_dir(), 'img'))
            abs_img = os.path.abspath(input_img)
            if not abs_img.startswith(img_dir) or not os.path.isfile(abs_img):
                return jsonify({'error': 'Invalid input image path'}), 400
        try:
            result = app_state.start_loop(json_file, output_folder, input_img=input_img)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/reset', methods=['POST'])
    def api_reset():
        app_state.stop_loop()
        return jsonify({'status': 'stopped'})

    @app.route('/api/workflows/clone', methods=['POST'])
    def api_clone_workflow():
        data = request.get_json()
        source = data.get('source')
        new_name = data.get('name', '').strip()
        content = data.get('content')  # optional edited JSON content

        if not new_name:
            return jsonify({'error': 'name is required'}), 400
        if not source and content is None:
            return jsonify({'error': 'source or content is required'}), 400

        # Sanitize name
        new_name = re.sub(r'[^\w\-.]', '_', new_name)
        if not new_name.endswith('.json'):
            new_name += '.json'

        data_dir = _get_data_dir()
        user_dir = os.path.join(data_dir, 'user')
        os.makedirs(user_dir, exist_ok=True)
        dest_path = os.path.join(user_dir, new_name)

        if os.path.exists(dest_path):
            return jsonify({'error': f'File already exists: {new_name}'}), 409

        if content is not None:
            # Validate JSON
            try:
                json_mod.loads(_strip_json_comments(content))
            except json_mod.JSONDecodeError as e:
                return jsonify({'error': f'Invalid JSON: {e}'}), 400
            with open(dest_path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            # Find source in data/ or data/user/
            source_basename = os.path.basename(source)
            source_path = os.path.join(data_dir, source_basename)
            if not os.path.isfile(source_path):
                source_path = os.path.join(user_dir, source_basename)
            if not os.path.isfile(source_path):
                return jsonify({'error': f'Source file not found: {source}'}), 404
            shutil.copy2(source_path, dest_path)

        return jsonify({'status': 'ok', 'name': new_name, 'path': dest_path})

    @app.route('/api/workflows/read', methods=['POST'])
    def api_read_workflow():
        """Read the raw content of a workflow JSON file for editing."""
        data = request.get_json()
        path = data.get('path')
        if not path or not os.path.isfile(path):
            return jsonify({'error': 'File not found'}), 404
        # Verify it's within the data directory
        data_dir = os.path.abspath(_get_data_dir())
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(data_dir):
            return jsonify({'error': 'Access denied'}), 403
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({'content': content})

    @app.route('/api/workflows/save', methods=['POST'])
    def api_save_workflow():
        """Save edits to an existing user workflow."""
        data = request.get_json()
        path = data.get('path')
        content = data.get('content')
        if not path or not content:
            return jsonify({'error': 'path and content are required'}), 400
        # Only allow saving to data/user/
        data_dir = os.path.abspath(_get_data_dir())
        user_dir = os.path.abspath(os.path.join(data_dir, 'user'))
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(user_dir):
            return jsonify({'error': 'Can only edit user workflows'}), 403
        if not os.path.isfile(abs_path):
            return jsonify({'error': 'File not found'}), 404
        try:
            json_mod.loads(_strip_json_comments(content))
        except json_mod.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON: {e}'}), 400
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'status': 'ok'})

    @app.route('/api/workflows', methods=['DELETE'])
    def api_delete_workflow():
        """Delete a user workflow."""
        data = request.get_json()
        path = data.get('path')
        if not path:
            return jsonify({'error': 'path is required'}), 400
        data_dir = os.path.abspath(_get_data_dir())
        user_dir = os.path.abspath(os.path.join(data_dir, 'user'))
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(user_dir):
            return jsonify({'error': 'Can only delete user workflows'}), 403
        if not os.path.isfile(abs_path):
            return jsonify({'error': 'File not found'}), 404
        os.remove(abs_path)
        return jsonify({'status': 'ok'})

    # --- Image management ---

    def _get_img_dir():
        return os.path.join(_get_data_dir(), 'img')

    @app.route('/api/images')
    def api_images():
        """List available starting images."""
        img_dir = _get_img_dir()
        gen_dir = os.path.join(img_dir, 'user')
        images = []

        def _scan_images(directory, source_label):
            if not os.path.isdir(directory):
                return
            for f in sorted(os.listdir(directory)):
                if os.path.splitext(f)[1].lower() not in IMAGE_EXTENSIONS:
                    continue
                full_path = os.path.join(directory, f)
                if os.path.isfile(full_path):
                    images.append({'name': f, 'path': full_path, 'source': source_label})

        _scan_images(img_dir, 'sample')
        _scan_images(gen_dir, 'uploaded')
        return jsonify({'images': images})

    @app.route('/api/images/upload', methods=['POST'])
    def api_upload_image():
        """Upload an image to data/img/user/."""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        f = request.files['file']
        if not f.filename:
            return jsonify({'error': 'No filename'}), 400
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            return jsonify({'error': f'Invalid image type: {ext}'}), 400
        filename = secure_filename(f.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
        gen_dir = os.path.join(_get_img_dir(), 'user')
        os.makedirs(gen_dir, exist_ok=True)
        dest = os.path.join(gen_dir, filename)
        f.save(dest)
        return jsonify({'name': filename, 'path': dest})

    @app.route('/api/images/file')
    def api_image_file():
        """Serve an image file for preview."""
        path = request.args.get('path')
        if not path:
            return jsonify({'error': 'path required'}), 400
        img_dir = os.path.abspath(_get_img_dir())
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(img_dir) or not os.path.isfile(abs_path):
            return jsonify({'error': 'Not found'}), 404
        return send_file(abs_path)

    # --- Loop status ---

    @app.route('/api/status')
    def api_status():
        state = _require_loop_state()
        if state is None:
            return jsonify({
                'status': 'stopped',
                'current_iteration': 0,
                'total_iterations': 0,
                'latest_image_index': 0,
                'error': None,
                'json_name': None,
                'has_input_image': True,
            })
        return jsonify({
            'status': state.get_status().value,
            'current_iteration': state.get_current_iteration(),
            'total_iterations': state.get_total_iterations(),
            'latest_image_index': state.get_latest_image_index(),
            'error': state.get_error(),
            'warning': state.get_warning(),
            'json_name': app_state.get_current_json_name(),
            'has_input_image': state.has_input_image(),
        })

    @app.route('/api/scenes')
    def api_scenes():
        state = _require_loop_state()
        if state is None:
            return jsonify({'scenes': []})
        sm = state.get_settings_manager()
        if sm is None:
            return jsonify({'scenes': []})
        scenes = []
        for ls in sm.workflow.all_settings:
            scenes.append({'start': ls.offset + 1, 'length': ls.loop_iterations})
        return jsonify({'scenes': scenes})

    @app.route('/api/image/<int:index>')
    def api_image(index: int):
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        image_store = app_state.get_image_store()
        if image_store is None:
            return jsonify({'error': 'No active loop'}), 503
        filename = _loop_img_filename(index)
        if image_store.has_image(filename):
            image_bytes = image_store.read_image_bytes(filename)
            return send_file(io.BytesIO(image_bytes), mimetype='image/png')
        else:
            return jsonify({'error': f'Image {index} not found'}), 404

    @app.route('/api/settings/<int:index>')
    def api_settings(index: int):
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        if index == 0:
            return jsonify({'settings': 'Starting image (no iteration settings)'})
        iteration = index - 1
        settings_json = state.get_settings(iteration)
        if settings_json is not None:
            return jsonify({'settings': settings_json})
        else:
            return jsonify({'error': f'Settings for image {index} not available'}), 404

    @app.route('/api/progress')
    def api_progress():
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        return jsonify(state.get_progress_info())

    @app.route('/api/pause', methods=['POST'])
    def api_pause():
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        state.pause()
        return jsonify({'status': 'paused'})

    @app.route('/api/resume', methods=['POST'])
    def api_resume():
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        state.resume()
        return jsonify({'status': 'resumed'})

    @app.route('/api/restart', methods=['POST'])
    def api_restart():
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        data = request.get_json()
        from_image_index = data.get('from_image_index')
        if from_image_index is None or from_image_index < 1:
            return jsonify({'error': 'from_image_index must be >= 1'}), 400

        state.request_restart(from_image_index)
        return jsonify({'status': 'restart_requested', 'from_image_index': from_image_index})

    @app.route('/api/export', methods=['POST'])
    def api_export():
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        if state.get_export_status() == 'running':
            return jsonify({'error': 'Export already in progress'}), 409

        data = request.get_json()
        fmt = data.get('format', 'gif')
        if fmt not in ('gif', 'mp4'):
            return jsonify({'error': 'format must be gif or mp4'}), 400

        filename = data.get('filename', 'export')
        filename = re.sub(r'[^\w\-]', '_', filename)
        if not filename:
            filename = 'export'

        params = {}
        params['frame_delay'] = str(data.get('frame_delay', 250))
        params['max_dim'] = str(data.get('max_dim', 768))

        start_frame = data.get('start_frame', 0)
        end_frame = data.get('end_frame', -1)
        if start_frame > 0:
            params['start_frame'] = str(start_frame)
        if end_frame >= 0:
            params['end_frame'] = str(end_frame)

        if fmt == 'mp4':
            params['v_bitrate'] = str(data.get('v_bitrate', '4000k'))
            mp3_file = data.get('mp3_file', '')
            if mp3_file:
                params['mp3_file'] = mp3_file

        if data.get('bounce', False):
            params['bounce'] = 'true'
            params['bounce_frame_skip'] = str(data.get('bounce_frame_skip', 0))

        output_path = os.path.join(state.get_output_folder(), f'{filename}.{fmt}')

        state.set_export_file(output_path)
        state.set_export_error(None)
        state.set_export_status('running')

        image_store = app_state.get_image_store()

        def run_export():
            try:
                anim_folder, needs_cleanup = image_store.get_paths_for_animation()
                try:
                    make_animation(
                        type=fmt,
                        input_folder=anim_folder,
                        output_animation=output_path,
                        params=params
                    )
                finally:
                    if needs_cleanup:
                        shutil.rmtree(anim_folder, ignore_errors=True)
                state.set_export_status('done')
            except Exception as e:
                state.set_export_error(str(e))
                state.set_export_status('error')

        threading.Thread(target=run_export, daemon=True).start()
        return jsonify({'status': 'started'})

    @app.route('/api/export/status')
    def api_export_status():
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        return jsonify({
            'status': state.get_export_status(),
            'error': state.get_export_error(),
        })

    @app.route('/api/export/download')
    def api_export_download():
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        if state.get_export_status() != 'done':
            return jsonify({'error': 'No export ready for download'}), 404
        filepath = state.get_export_file()
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Export file not found'}), 404

        basename = os.path.basename(filepath)
        state.clear_export()
        return send_file(filepath, as_attachment=True, download_name=basename)

    # --- Settings overrides ---

    OVERRIDE_ALLOWED_FIELDS = {
        'prompt', 'neg_prompt', 'denoise_amt', 'denoise_steps', 'cfg', 'seed',
        'checkpoint', 'loras', 'transforms', 'canny', 'con_deltas',
    }

    def _serialize_setting_val(val):
        """Convert a LoopSettings field value to JSON-serializable form."""
        if val is EMPTY_OBJECT or (isinstance(val, list) and val == EMPTY_LIST):
            return None
        return serialize_override_value(val)

    def _deserialize_override(key, value):
        """Convert a JSON override value to the appropriate Python type for setattr."""
        return deserialize_override_value(key, value)

    @app.route('/api/settings/<int:index>/raw')
    def api_settings_raw(index: int):
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        if index == 0:
            return jsonify({'error': 'No settings for starting image'}), 400
        iteration = index - 1
        sm = state.get_settings_manager()
        if sm is None:
            return jsonify({'error': 'Settings manager not ready'}), 503
        section_idx, loopsetting = sm.get_loopsettings_for_iter(iteration)
        raw = {}
        for field_name in OVERRIDE_ALLOWED_FIELDS:
            raw[field_name] = _serialize_setting_val(getattr(loopsetting, field_name))
        raw['section_idx'] = section_idx
        raw['loop_iterations'] = loopsetting.loop_iterations
        raw['offset'] = loopsetting.offset
        return jsonify({'raw_settings': raw})

    @app.route('/api/override/frame', methods=['POST'])
    def api_override_frame():
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        if state.get_status() != LoopStatus.PAUSED:
            return jsonify({'error': 'Pause the loop before applying overrides'}), 409
        data = request.get_json()
        iteration = data.get('iteration')
        overrides_raw = data.get('overrides', {})
        if iteration is None:
            return jsonify({'error': 'iteration required'}), 400
        overrides = {}
        for key, value in overrides_raw.items():
            if key not in OVERRIDE_ALLOWED_FIELDS:
                return jsonify({'error': f'Field {key} not allowed'}), 400
            overrides[key] = _deserialize_override(key, value)
        try:
            state.apply_frame_override(iteration, overrides)
        except KeyError as e:
            return jsonify({'error': str(e)}), 400
        state.persist_overrides()
        return jsonify({'status': 'ok', 'iteration': iteration, 'overrides': list(overrides.keys())})

    @app.route('/api/override/formula', methods=['POST'])
    def api_override_formula():
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        if state.get_status() != LoopStatus.PAUSED:
            return jsonify({'error': 'Pause the loop before applying overrides'}), 409
        data = request.get_json()
        iteration = data.get('iteration')
        overrides = data.get('overrides', {})
        if iteration is None:
            return jsonify({'error': 'iteration required'}), 400
        sm = state.get_settings_manager()
        if sm is None:
            return jsonify({'error': 'Settings manager not ready'}), 503
        _section_idx, loopsetting = sm.get_loopsettings_for_iter(iteration)
        formula_tracked = {}
        for key, value in overrides.items():
            if key not in OVERRIDE_ALLOWED_FIELDS:
                return jsonify({'error': f'Field {key} not allowed'}), 400
            # For formula mode: EXPR fields keep strings as expressions,
            # complex types get deserialized to proper objects
            if key in ('loras', 'canny', 'con_deltas', 'transforms'):
                deserialized = _deserialize_override(key, value)
                setattr(loopsetting, key, deserialized)
                formula_tracked[key] = deserialized
            else:
                # Strings stay as strings (expressions), primitives pass through
                setattr(loopsetting, key, value)
                formula_tracked[key] = value
        state.record_formula_override(_section_idx, formula_tracked)
        state.re_elaborate_from(iteration, sm)
        state.persist_overrides()
        return jsonify({'status': 'ok', 'section_idx': _section_idx})

    @app.route('/api/overrides')
    def api_get_overrides():
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        return jsonify({
            'overridden_frames': state.get_all_overridden_frames(),
            'overridden_sections': state.get_all_overridden_sections(),
        })

    @app.route('/api/override/reset', methods=['POST'])
    def api_override_reset():
        state = _require_loop_state()
        if state is None:
            return jsonify({'error': 'No active loop'}), 503
        if state.get_status() == LoopStatus.RUNNING:
            return jsonify({'error': 'Pause the loop before resetting overrides'}), 409
        json_file = state.get_json_file()
        animation_params = state.get_animation_params()
        if json_file is None:
            return jsonify({'error': 'No JSON file available for reset'}), 500
        # Create a fresh SettingsManager from the original JSON (outside the
        # LoopState lock — SM construction & validation can be slow).
        sm = SettingsManager(json_file, animation_params)
        sm.validate()
        state.set_settings_manager(sm)
        state.reset_all_overrides(sm)
        state.delete_overrides_file()
        return jsonify({'status': 'reset'})

    return app
