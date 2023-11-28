import os

import PySimpleGUI as sg

from src.in_out import is_valid_path
from src.service.gui_service import MainService


class GUIView:

    def __init__(self, service: MainService):
        sg.theme('Reddit')  # Add a touch of color

        # All the stuff inside your window.
        # init window
        self.layout = [
            [sg.Text('选择输出文件名;')],
            [sg.InputText(key='-OUTPUT-', default_text=os.path.join(os.getcwd(), '输出.xlsx'))],
            [sg.Text('选择引入模版文件:')],
            [sg.InputText(key='-INPUT-'), sg.FileBrowse(button_text="选择文件",
                                                        file_types=(('Excel Files', '*.xlsx'),
                                                                    ('Excel Files', '*.xls')),
                                                        initial_folder=os.getcwd()), ],
            [sg.Text('选择操作力计算数据表文件:')],
            [sg.InputText(key='-OPERATION-',
                          default_text=os.path.join(os.getcwd(), "操作力计算数据表.xlsx")),
             sg.FileBrowse(button_text="选择文件",
                           file_types=(('Excel Files', '*.xlsx'),
                                       ('Excel Files', '*.xls')),
                           initial_folder=os.getcwd()), ],
            [sg.Text('选择标准格式文件:')],
            [sg.InputText(key='-STANDARD-',
                          default_text=os.path.join(os.getcwd(),"标准格式.xlsx")),
             sg.FileBrowse(button_text="选择文件",
                           file_types=(('Excel Files', '*.xlsx'),
                                       ('Excel Files', '*.xls')),
                           initial_folder=os.getcwd())],
            [sg.Text('选择阀门特征系数:')],
            [sg.InputText(key='-VALVE-CHARACTERISTIC-',
                          default_text=os.path.join(os.getcwd(), "阀门特征系数表.xlsx")),
             sg.FileBrowse(button_text="选择文件",
                           file_types=(('Excel Files', '*.xlsx'),
                                       ('Excel Files', '*.xls')),
                           initial_folder=os.getcwd())],
            [sg.Text('选择cv表:')],
            [sg.InputText(key='-CV-',
                          default_text=os.path.join(os.getcwd(), "Cv值表.xlsx")),
             sg.FileBrowse(button_text="选择文件",
                           file_types=(('Excel Files', '*.xlsx'),
                                       ('Excel Files', '*.xls')),
                           initial_folder=os.getcwd())],
            [sg.Button('开始'), sg.Button('取消')]
        ]

        self.window = sg.Window('表单函数', self.layout, finalize=True)
        self.service = service
        self.output_window = sg.Window('正在导出', [[sg.Text('正在导出，请稍等')]])
        self.progress_window = None
        self.start_window()

    @property
    def input(self):
        if not is_valid_path(self.window['-INPUT-'].get(), '.xlsx'):
            raise ValueError('Invalid input file or path does not exist')
        return self.window['-INPUT-'].get()

    @property
    def operation(self):
        if not is_valid_path(self.window['-OPERATION-'].get(), '.xlsx'):
            raise ValueError('Invalid operation file or path does not exist')
        return self.window['-OPERATION-'].get()

    @property
    def standard(self):
        if not is_valid_path(self.window['-STANDARD-'].get(), '.xlsx'):
            raise ValueError('Invalid standard file or path does not exist')
        return self.window['-STANDARD-'].get()

    @property
    def valve_characteristic(self):
        if not is_valid_path(self.window['-VALVE-CHARACTERISTIC-'].get(), '.xlsx'):
            raise ValueError('Invalid valve characteristic file or path does not exist')
        return self.window['-VALVE-CHARACTERISTIC-'].get()

    @property
    def cv_path(self):
        if not is_valid_path(self.window['-CV-'].get(), '.xlsx'):
            raise ValueError('Invalid cv file or path does not exist')
        return self.window['-CV-'].get()

    @property
    def output(self):
        return self.window['-OUTPUT-'].get()

    def start_window(self):
        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED or event == '取消':
                break
            if event == '开始':
                try:
                    self.on_start_clicked()
                except Exception as e:
                    sg.popup_error(e)
                    continue
        self.window.close()
        return event, values

    @property
    def task_size(self):
        return self.service.task_size

    def update_progress(self, progress):
        self.progress_window['-PROGRESS-'].update_bar(progress)

    def on_start_clicked(self):
        # attach input, operation, standard, valve_characteristic file to service
        self.service.attach(self)
        progress_layout = [[sg.Text('任务完成进度')],
                           [sg.ProgressBar(self.task_size, orientation='h', size=(20, 20), key='-PROGRESS-')],
                           [sg.Cancel()]]
        self.progress_window = sg.Window('任务进度', progress_layout)

        progress = 0
        while not self.service.process_end():
            event, values = self.progress_window.read(timeout=10)
            if event == sg.WIN_CLOSED or event == 'Cancel':
                break
            self.service.process()
            self.service.next()
            self.update_progress(progress)
            progress += 1
        self.progress_window.close()
        if progress == self.task_size:
            # 弹窗提示导出
            self.output_window.read(timeout=10)
            self.service.export(self.output)
            self.output_window.close()
            sg.popup_ok('处理完成')
