import tensorflow as tf
from donut import DonutTrainer, DonutPredictor
from donut.demo.donut_model import get_model


def train_prediction(train_values, train_labels, train_missing, test_values, test_missing, mean, std):
    model, model_vs = get_model()
    trainer = DonutTrainer(model=model, model_vs=model_vs)
    predictor = DonutPredictor(model)
    with tf.Session().as_default():
        trainer.fit(values=train_values, labels=train_labels, missing=train_missing, mean=mean, std=std)
        return predictor.get_score(test_values, test_missing)
