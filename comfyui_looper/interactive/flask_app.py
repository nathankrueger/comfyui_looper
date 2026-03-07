import os
from flask import Flask, jsonify, request, send_file, send_from_directory
from interactive.loop_state import LoopState, LoopStatus


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

    return app
