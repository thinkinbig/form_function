import os

from src.exceptions import UploadFileError
from src.service.gui_service import MainService


class RestService:

    def __init__(self, service: MainService):
        self.config = None
        self.service = service

    def set_config(self, config):
        self.config = config

    @property
    def is_uploaded(self):
        if self.config is None:
            return False
        return os.path.exists(self.config['INPUT']) and \
            os.path.exists(self.config['OPERATION']) and \
            os.path.exists(self.config['STANDARD']) and \
            os.path.exists(self.config['VALVE-CHARACTERISTIC']) and \
            os.path.exists(self.config['CV'])

    @property
    def input(self):
        if not self.is_uploaded:
            raise UploadFileError("文件未全部上传")
        return self.config['INPUT']

    @property
    def operation(self):
        if not self.is_uploaded:
            raise UploadFileError("文件未全部上传")
        return self.config['OPERATION']

    @property
    def standard(self):
        if not self.is_uploaded:
            raise UploadFileError("文件未全部上传")
        return self.config['STANDARD']

    @property
    def valve_characteristic(self):
        if not self.is_uploaded:
            raise UploadFileError("文件未全部上传")
        return self.config['VALVE-CHARACTERISTIC']

    @property
    def cv_path(self):
        if not self.is_uploaded:
            raise UploadFileError("文件未全部上传")
        return self.config['CV']

    @property
    def output(self):
        return self.config['OUTPUT']

    @property
    def task_size(self):
        return self.service.task_size

    def run(self):
        self.service.attach(self)
        while not self.service.process_end():
            self.service.process()
            self.service.next()
        self.service.export(self.output)
