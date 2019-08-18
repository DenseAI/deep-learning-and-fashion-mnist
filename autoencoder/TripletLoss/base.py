class TrainerBase(object):
    """
    训练器基类
    """

    def __init__(self, model, data, config, class_mode=None):
        self.model = model  # 模型
        self.data = data  # 数据
        self.config = config  # 配置
        self.class_mode = class_mode

    def train(self):
        """
        训练逻辑
        """
        raise NotImplementedError


class ModelBase(object):
    """
    模型基类
    """

    def __init__(self, config=None):
        self.config = config  # 配置
        self.model = None  # 模型
        self.class_model = None  # 模型

    def save(self, checkpoint_path):
        """
        存储checkpoint, 路径定义于配置文件中
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Saving model...")
        self.model.save_weights(checkpoint_path)
        print("[INFO] Model saved")

    def save_class_model(self, checkpoint_path):
        """
        存储checkpoint, 路径定义于配置文件中
        """
        if self.class_model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Saving model...")
        self.class_model.save_weights(checkpoint_path)
        print("[INFO] Model saved")

    def load(self, checkpoint_path):
        """
        加载checkpoint, 路径定义于配置文件中
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("[INFO] Model loaded")

    def load_class_model(self, checkpoint_path):
        """
        加载checkpoint, 路径定义于配置文件中
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("[INFO] Model loaded")

    def build_model(self):
        """
        构建模型
        """
        raise NotImplementedError



class DataLoaderBase(object):
    """
    数据加载的基类
    """

    def __init__(self, config):
        self.config = config  # 设置配置信息

    def get_train_data(self):
        """
        获取训练数据
        """
        raise NotImplementedError

    def get_test_data(self):
        """
        获取测试数据
        """
        raise NotImplementedError