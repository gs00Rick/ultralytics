import os

ORIGIN_DATASETS_PATH = {
    "tt_100k": "C:/Users/Administrator/Desktop/ObjectDetection/tt100k_2021/"
}


class FilePathUtils:
    def __init__(self):
        pass

    @staticmethod
    def get_origin_datasets_paths(name):
        """
        根据数据集名称获取原始数据集的路径

        参数:
        name(str): 数据集的名称

        返回:
        str: 原始数据集的路径
        """
        return ORIGIN_DATASETS_PATH[name]

    @staticmethod
    def get_project_root():
        """
        获取项目根目录的路径。

        这个函数用于找到当前项目的根目录路径，对于多文件和模块的项目结构，非常有用。

        Returns:
            str: 项目根目录的绝对路径。
        """
        # 获取当前文件的绝对路径
        abs_path = os.path.abspath(__file__)

        # 获取当前文件所在的目录
        current_dir = os.path.dirname(abs_path)

        # 向上回溯到项目根目录（可以根据实际项目结构调整回溯的层数）
        project_root = current_dir
        while not os.path.exists(os.path.join(project_root, 'README.md')):
            project_root = os.path.dirname(project_root)

        return project_root




if __name__ == '__main__':
    print(FilePathUtils.get_project_root())
