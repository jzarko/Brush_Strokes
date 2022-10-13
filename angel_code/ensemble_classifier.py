import numpy as np
import tensorflow as tf
import random
import os

from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
from tqdm import tqdm

class EnsembleClassifier:

    model_names = ['angel_kanji_model', 'john_kanji_model', 'justin_kanji_model']
    trained_models = {}
    missclassifications = {}


    def __init__(self, dir) -> None:
        working_dir = os.path.join(os.getcwd(), dir)

        for name in self.model_names:
            model_dir = os.path.join(working_dir, name)
            if os.path.isdir(model_dir):
                self.trained_models[name] = models.load_model(model_dir)
                print(f'Loaded model: {name}')
            else:
                print(f"Invalid model name {name}")
        pass

    def predict(self, element):
        model_predictions = [None] * len(self.trained_models.keys())
        ensemble_predictions = {}

        # Use each individual model to predict
        for i, (name, model) in enumerate(self.trained_models.items()):
            model_predictions[i] = np.argmax(model.predict(element))

        # Vote on final output
        for pred in model_predictions:
            if pred in ensemble_predictions.keys():
                ensemble_predictions[pred] += 1
            else:
                ensemble_predictions[pred] = 1
        
        # print(f'True label:\t\t{label}')

        prediction = None

        # All models agree
        if len(ensemble_predictions) == 1:
            prediction = list(ensemble_predictions.keys())[0]
            # print(f'Ensemble pred. label:\t{prediction}')
        
        # 1 model disagrees
        elif len(ensemble_predictions) == 2:
            prediction = max(ensemble_predictions, key=ensemble_predictions.get)
            # print(f'Ensemble pred. label:\t{prediction}')

        # all models disagree
        # need to return model with best track record
        else:
            prediction = random.choice(list(ensemble_predictions.keys()))
            # print(f'Ensemble pred. label:\t{prediction}')

        return prediction
        

    def validate(self, validation_data):
        for i, (element, label) in tqdm(enumerate(validation_data)):
            true_label = validation_data.class_names[label[0].numpy()]
            prediction = self.predict(element)
            pred_label = validation_data.class_names[prediction]
            if pred_label != true_label:
                # print(f'Model confused: {pred_label} for {true_label}')

                if pred_label in self.missclassifications.keys():
                    self.missclassifications[pred_label] += 1
                else:
                    self.missclassifications[pred_label] = 1
            

    def demo(self, validation_data):
        for i, (element, label) in enumerate(validation_data):
            true_label = validation_data.class_names[label[0].numpy()]
            prediction = self.predict(element)
            pred_label = validation_data.class_names[prediction]
            print(f'PRED: {pred_label}')
            print(f'TRUE: {true_label}')
            if pred_label != true_label:
                print(f'Model confused: {pred_label} for {true_label}')

            self.plot_image(element.numpy()[0].astype("uint8"), true_label)

            if i >= 4:
                break
    

    def plot_image(self, image, label):
        plt.figure(figsize=(1, 2))
        plt.imshow(image)
        plt.title(label)
        plt.axis("off")
        plt.show()
        

    def plot_missclassifications(self):
        plt.bar(self.missclassifications.keys(), 
                self.missclassifications.values(), 
                1.0, color='r')
        plt.show()



if __name__ == "__main__":

    new_kkanji_midterm_dataset_val = tf.keras.utils.image_dataset_from_directory(
                                        '.\\Code\\midterm_dataset\\',
                                        validation_split=0.3,
                                        subset="validation",
                                        seed=132,
                                        image_size=(64, 64),
                                        batch_size=1)

    my_ensemble_model = EnsembleClassifier('.\\Code')
    my_ensemble_model.demo(new_kkanji_midterm_dataset_val)
    print()
    # my_ensemble_model.validate(new_kkanji_midterm_dataset_val)
    # 95.63 accuracy 