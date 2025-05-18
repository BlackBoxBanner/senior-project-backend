import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from module.ColorModule import ColorModule
from module.UIComponentDetector import UIComponentDetector
from module.LayoutAnalyzer import LayoutAnalyzer, AxisEnum

app = Flask(__name__)

# ==== CORS SETUP ====
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def hello_world():
    return { 'hello': 'world' }

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({ 'message': 'No image part in the request' }), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({ 'message': 'No image selected for uploading' }), 400

    # Read the raw bytes and Base64-encode
    file_data = file.read()
    encoded_str = base64.b64encode(file_data).decode('utf-8')

    
    ui_detector = UIComponentDetector()
    detected_ui = ui_detector.detect_from_base64(encoded_str)

    cm = ColorModule(encoded_str, detections=detected_ui)
    la = LayoutAnalyzer(detected_ui, tol_x=20, tol_y=20)

    # Generate grid and collect skipped detections
    image_grid, skipped_detections = la.generate_grid_with_skipped(
        tol_x=20, tol_y=20, allow_multi_assign=True, debug=False, allow_overlaps=True
    )

    row_count = len(image_grid)
    col_count = len(image_grid[0]) if image_grid else 0

    # Misalignment details (actual misalign + skipped)
    row_misaligned, row_skipped = la.get_misaligned_and_skipped(AxisEnum.HORIZONTAL)
    col_misaligned, col_skipped = la.get_misaligned_and_skipped(AxisEnum.VERTICAL)

    h_score = la.calculate_misalignment_percentage(AxisEnum.HORIZONTAL)
    v_score = la.calculate_misalignment_percentage(AxisEnum.VERTICAL)


    # Color analysis
    color_content = cm.extract_dominant_colors()
    color_content_percent = cm.calculate_percentages(color_content)
    color_contrast = cm.contrast_ratio(
        color1=color_content[0][0],
        color2=color_content[1][0]
    )

    return jsonify({
        'row':{
            'count': row_count,
            'misaligned': row_misaligned,
            'skipped': row_skipped,
            'misaligned': h_score
        },
        'col':{
            'count': col_count,
            'misaligned': col_misaligned,
            'skipped': col_skipped,
            'misaligned': v_score
        },
        'color': {
            'dominant_colors': color_content,
            'percentages': color_content_percent,
            'contrast_ratio': color_contrast
        },
        'skipped_detections': skipped_detections,
        'image': f"data:{file.mimetype};base64,{encoded_str}",
    }), 200

if __name__ == "__main__":
    app.run(debug=(os.getenv("FLASK_ENV")=="development"))