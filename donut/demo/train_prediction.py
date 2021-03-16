import tensorflow as tf
from donut import DonutTrainer, DonutPredictor
from donut.demo.donut_model import get_model


def train_prediction(train_values, train_labels, train_missing, test_values, test_missing, mean, std):
    model, model_vs = get_model()
    trainer = DonutTrainer(model=model, model_vs=model_vs)
    predictor = DonutPredictor(model)
    with tf.Session().as_default():
        trainer.fit(train_values, train_labels, train_missing, mean, std)
        test_score = predictor.get_score(test_values, test_missing)
    return test_score
