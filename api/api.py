
import os
import logging
FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)

from flask import Flask, send_from_directory
from flask_cors import CORS
from api.controller import images, train, predict, models
from api.util.const import Const

static_folder = 'static'
results_folder = 'results'
static_folder_abs = os.path.abspath(static_folder)
app = Flask(__name__, static_folder=static_folder_abs, static_url_path='/static')
# prepare static folder
os.makedirs(static_folder_abs, exist_ok=True)
try:
    os.symlink(
        os.path.abspath(results_folder),
        os.path.join(static_folder_abs, results_folder)
    )
except FileExistsError:
    pass

cors = CORS(app, origins="*")

app.register_blueprint(images.app, name='images', url_prefix=Const.version_prefix)
app.register_blueprint(train.app, name='train', url_prefix=Const.version_prefix)
app.register_blueprint(models.app, name='models', url_prefix=Const.version_prefix)
app.register_blueprint(predict.app, url_prefix = Const.version_prefix)

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0')
