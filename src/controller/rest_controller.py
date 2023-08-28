import logging
import os

from src.service.gui_service import MainService
from src.service.rest_service import RestService
from src.utils import allowed_file

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

from flask import Flask, request, send_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

rest_service = RestService(MainService())

@app.route('/')
def it_works():
    return 'It works!'


@app.route('/formFunction', methods=['POST'])
def form_function():
    logging.info('formFunction')
    if len(request.form) == 0 or len(request.files) == 0:
        logging.debug('No form or file part')
        return 'No form or file part'
    if 'name' not in request.form:
        logging.debug('No file part name')
        return 'No file part name'
    if 'input' not in request.files:
        logging.debug('No file part input')
        return 'No file part input'
    if 'operation' not in request.files:
        logging.debug('No file part operation')
        return 'No file part operation'
    if 'standard' not in request.files:
        logging.debug('No file part standard')
        return 'No file part standard'
    if 'valve_characteristic' not in request.files:
        logging.debug('No file part valve_characteristic')
        return 'No file part valve_characteristic'
    if 'cv' not in request.files:
        logging.debug('No file part cv')
        return 'No file part cv'
    name = request.form['name']
    files = request.files
    input = files['input']
    operation = files['operation']
    standard = files['standard']
    valve_characteristic = files['valve_characteristic']
    cv = files['cv']
    if input.filename.strip() != ' ' and not allowed_file(input.filename):
        logging.debug('Invalid input file')
        return 'Invalid input file'
    if operation.filename.strip() != ' ' and not allowed_file(operation.filename):
        logging.debug('Invalid operation file')
        return 'Invalid operation file'
    if standard.filename.strip() != '' and not allowed_file(standard.filename):
        logging.debug('Invalid standard file')
        return 'Invalid standard file'
    if valve_characteristic.filename != ' ' and not allowed_file(valve_characteristic.filename):
        logging.debug('Invalid valve_characteristic file')
        return 'Invalid valve_characteristic file'
    if cv.filename.strip() != '' and not allowed_file(cv.filename):
        logging.debug('Invalid cv file')
        return 'Invalid cv file'

    # save files to local storage
    input_filename = input.filename
    operation_filename = operation.filename
    standard_filename = standard.filename
    valve_characteristic_filename = valve_characteristic.filename
    cv_filename = cv.filename

    config = {
        'INPUT': os.path.join(app.config['UPLOAD_FOLDER'], input_filename),
        'OPERATION': os.path.join(app.config['UPLOAD_FOLDER'], operation_filename),
        'STANDARD': os.path.join(app.config['UPLOAD_FOLDER'], standard_filename),
        'VALVE-CHARACTERISTIC': os.path.join(app.config['UPLOAD_FOLDER'], valve_characteristic_filename),
        'CV': os.path.join(app.config['UPLOAD_FOLDER'], cv_filename),
        'OUTPUT': os.path.join(app.config['UPLOAD_FOLDER'], name + '.xlsx'),
    }
    input.save(config['INPUT'])
    operation.save(config['OPERATION'])
    standard.save(config['STANDARD'])
    valve_characteristic.save(config['VALVE-CHARACTERISTIC'])
    cv.save(config['CV'])

    rest_service.set_config(config)
    try:
        rest_service.run()
    except Exception as e:
        return str(e)
    logging.info('formFunction finished')
    return send_file(config['OUTPUT'], as_attachment=True)



