import logging

from src.GUI import GUIView
from src.service import MainService


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        # filemode='w',
                        # filename='log.txt',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    service = MainService()
    gui = GUIView(service)
