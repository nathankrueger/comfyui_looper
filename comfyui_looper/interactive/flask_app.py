import os
import re
import threading
from flask import Flask, jsonify, request, send_file, send_from_directory
from interactive.loop_state import LoopState, LoopStatus
from image_processing.animator import make_animation


def _loop_img_filename(idx: int) -> str:
    return f"loop_img_{idx:06}.png"


def create_app(state: LoopState) -> Flask:
    app = Flask(__name__, static_folder='static', static_url_path='/static')

    @app.route('/')
    def index():
        return send_from_directory(app.static_folder, 'index.html')

    @app.route('/api/status')
    def api_status():
        return jsonify({
            'status': state.get_status().value,
            'current_iteration': state.get_current_iteration(),
            'total_iterations': state.get_total_iterations(),
            'latest_image_index': state.get_latest_image_index(),
            'error': state.get_error(),
        })

    @app.route('/api/image/<int:index>')
    def api_image(index: int):
        filename = _loop_img_filename(index)
        filepath = os.path.join(state.get_output_folder(), filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/png')
        else:
            return jsonify({'error': f'Image {index} not found'}), 404

    @app.route('/api/settings/<int:index>')
    def api_settings(index: int):
        if index == 0:
            return jsonify({'settings': 'Starting image (no iteration settings)'})
        iteration = index - 1
        settings_json = state.get_settings(iteration)
        if settings_json is not None:
            return jsonify({'settings': settings_json})
        else:
            return jsonify({'error': f'Settings for image {index} not available'}), 404

    @app.route('/api/pause', methods=['POST'])
    def api_pause():
        state.pause()
        return jsonify({'status': 'paused'})

    @app.route('/api/resume', methods=['POST'])
    def api_resume():
        state.resume()
        return jsonify({'status': 'resumed'})

    @app.route('/api/restart', methods=['POST'])
    def api_restart():
        data = request.get_json()
        from_image_index = data.get('from_image_index')
        if from_image_index is None or from_image_index < 1:
            return jsonify({'error': 'from_image_index must be >= 1'}), 400

        state.request_restart(from_image_index)
        # If paused, unblock the loop thread so it can process the restart
        if state.get_status() == LoopStatus.PAUSED:
            state.resume()

        return jsonify({'status': 'restart_requested', 'from_image_index': from_image_index})

    @app.route('/api/export', methods=['POST'])
    def api_export():
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

        def run_export():
            try:
                make_animation(
                    type=fmt,
                    input_folder=state.get_output_folder(),
                    output_animation=output_path,
                    params=params
                )
                state.set_export_status('done')
            except Exception as e:
                state.set_export_error(str(e))
                state.set_export_status('error')

        threading.Thread(target=run_export, daemon=True).start()
        return jsonify({'status': 'started'})

    @app.route('/api/export/status')
    def api_export_status():
        return jsonify({
            'status': state.get_export_status(),
            'error': state.get_export_error(),
        })

    @app.route('/api/export/download')
    def api_export_download():
        if state.get_export_status() != 'done':
            return jsonify({'error': 'No export ready for download'}), 404
        filepath = state.get_export_file()
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Export file not found'}), 404

        basename = os.path.basename(filepath)
        state.clear_export()
        return send_file(filepath, as_attachment=True, download_name=basename)

    return app
