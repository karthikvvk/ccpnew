from flask import Flask, send_file, abort
import os

from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


BASE_VIDEO_DIR = os.path.abspath("videos")
print(BASE_VIDEO_DIR)

@app.route("/api/v1/<jobid>/download/video", methods=["GET"])
def download_video(jobid):
    # Explicit mapping: jobid -> filename
    video_filename = f"{jobid}.mp4"
    video_path = os.path.join(BASE_VIDEO_DIR, video_filename)

    # Prevent path traversal
    if not os.path.commonpath([BASE_VIDEO_DIR, video_path]) == BASE_VIDEO_DIR:
        abort(403)

    if not os.path.exists(video_path):
        abort(404, description="Video not found")

    return send_file(
        video_path,
        mimetype="video/mp4",
        as_attachment=False,
        conditional=True  # enables HTTP range requests
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
