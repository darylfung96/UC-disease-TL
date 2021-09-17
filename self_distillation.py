from abc import ABC, abstractmethod
import torch.nn as nn
import torch.nn.functional as F


class LightningDistillation(ABC):
    @abstractmethod
    def training_step(self, loss):
        pass

    @abstractmethod
    def validation_step(self, args):
        """

        :param args:
        :return:
        """
        pass


class ModelDistillation(ABC):
    @abstractmethod
    def model_init(self, args):
        pass

    @abstractmethod
    def get_self_distillation_loss(self, args):
        pass

    @abstractmethod
    def model_distillation_forward(self, last_output):
        pass


class FirstLightningDistillation(LightningDistillation):

    def __init__(self):
        self.criterion = nn.BCELoss()
        self.model = FirstModelDistillation()
        self.alpha = 0.1
        self.beta = 1e-6

    def training_step(self, args):
        loss = args['loss']
        y = args['y']
        feature_loss, entropy_feature_loss = self.model.get_self_distillation_loss(args)
        post_output_layer1, post_output_layer2, \
        post_output_layer3, post_output_layer4 = self.model.get_sub_classifiers()
        sub_loss_1 = self.criterion(post_output_layer1, y)
        sub_loss_2 = self.criterion(post_output_layer2, y)
        sub_loss_3 = self.criterion(post_output_layer3, y)
        sub_loss_4 = self.criterion(post_output_layer4, y)

        entropy_loss = (1 - self.alpha) * (sub_loss_1 + loss + sub_loss_1 + sub_loss_2 + sub_loss_3 + sub_loss_4)
        entropy_feature_loss = self.alpha * entropy_feature_loss
        feature_loss = self.beta * feature_loss

        loss = entropy_loss + entropy_feature_loss + feature_loss
        return loss

    def validation_step(self, args):
        """

        :param args: loss, main_pre_output_layer_features, post_main_output_layer, y
        :return:
        """
        loss = args['loss']
        y = args['y']
        feature_loss, entropy_feature_loss = self.model.get_self_distillation_loss(args)
        post_output_layer1, post_output_layer2, \
        post_output_layer3, post_output_layer4 = self.model.get_sub_classifiers()
        sub_loss_1 = self.criterion(post_output_layer1, y)
        sub_loss_2 = self.criterion(post_output_layer2, y)
        sub_loss_3 = self.criterion(post_output_layer3, y)
        sub_loss_4 = self.criterion(post_output_layer4, y)

        entropy_loss = (1 - self.alpha) * (sub_loss_1 + loss + sub_loss_1 + sub_loss_2 + sub_loss_3 + sub_loss_4)
        entropy_feature_loss = self.alpha * entropy_feature_loss
        feature_loss = self.beta * feature_loss
        loss = entropy_loss + entropy_feature_loss + feature_loss
        return loss


class FirstModelDistillation(ModelDistillation):
    def __init__(self):
        self.feature_loss = nn.L1Loss()
        self.entropy_model_loss = nn.BCELoss()

    def get_sub_classifiers(self):
        return self.post_output_layer1, self.post_output_layer2, self.post_output_layer3, self.post_output_layer4

    def model_init(self, args):
        hidden_size = args['hidden_size']
        output_size = args['output_size']
        self.attention = args['attention']
        create_layer_block = args['create_layer_block']
        self.pre_output_layer1, self.output_layer1 = create_layer_block(hidden_size, output_size, self.attention)
        self.pre_output_layer2, self.output_layer2 = create_layer_block(hidden_size, output_size, self.attention)
        self.pre_output_layer3, self.output_layer3 = create_layer_block(hidden_size, output_size, self.attention)
        self.pre_output_layer4, self.output_layer4 = create_layer_block(hidden_size, output_size, self.attention)

    def get_self_distillation_loss(self, args):
        main_pre_output_layer_features = args['main_pre_output_layer_features']
        post_main_output_layer = args['post_main_output_layer']
        # 1
        feature_loss_1 = self.feature_loss(self.pre_output_layer1_features,
                                           main_pre_output_layer_features.detach())
        entropy_feature_loss_1 = self.entropy_model_loss(self.post_output_layer1, post_main_output_layer.detach())
        # 2
        feature_loss_2 = self.feature_loss(self.pre_output_layer2_features,
                                           main_pre_output_layer_features.detach())
        entropy_feature_loss_2 = self.entropy_model_loss(self.post_output_layer2, post_main_output_layer.detach())
        # 3
        feature_loss_3 = self.feature_loss(self.pre_output_layer3_features,
                                           main_pre_output_layer_features.detach())
        entropy_feature_loss_3 = self.entropy_model_loss(self.post_output_layer3, post_main_output_layer.detach())
        # 4
        feature_loss_4 = self.feature_loss(self.pre_output_layer4_features,
                                           main_pre_output_layer_features.detach())
        entropy_feature_loss_4 = self.entropy_model_loss(self.post_output_layer4, post_main_output_layer.detach())

        feature_losses = feature_loss_1 + feature_loss_2 + feature_loss_3 + feature_loss_4
        entropy_feature_losses = entropy_feature_loss_1 + entropy_feature_loss_2 + \
                                 entropy_feature_loss_3 + entropy_feature_loss_4

        return feature_losses, entropy_feature_losses

    def model_distillation_forward(self, last_output):
        # output 1
        self.pre_output_layer1_features = self.pre_output_layer1(last_output)
        output_layer1 = self.output_layer1(self.pre_output_layer1_features)
        self.post_output_layer1 = F.sigmoid(output_layer1)

        # output 2
        if self.attention:
            pre_output_layer1_features = last_output
        else:
            pre_output_layer1_features = self.pre_output_layer1_features
        self.pre_output_layer2_features = self.pre_output_layer2(pre_output_layer1_features)
        output_layer2 = self.output_layer2(self.pre_output_layer2_features)
        self.post_output_layer2 = F.sigmoid(output_layer2)

        # output 3
        if self.attention:
            pre_output_layer2_features = last_output
        else:
            pre_output_layer2_features = self.pre_output_layer2_features
        self.pre_output_layer3_features = self.pre_output_layer3(pre_output_layer2_features)
        output_layer3 = self.output_layer3(self.pre_output_layer3_features)
        self.post_output_layer3 = F.sigmoid(output_layer3)

        # output 4
        if self.attention:
            pre_output_layer3_features = last_output
        else:
            pre_output_layer3_features = self.pre_output_layer3_features
        self.pre_output_layer4_features = self.pre_output_layer4(pre_output_layer3_features)
        self.post_output_layer4 = F.sigmoid(self.output_layer4(self.pre_output_layer4_features))