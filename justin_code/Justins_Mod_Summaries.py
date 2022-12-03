# This file is the template architecture of the best performing base model and
#   top model combination. Transfer learning from K49 data allows the Kanji
#   classification to acheieves 93.5% accuracy for this model.

# Best base model implementation (optuna has not found an optimal base)
# val_accuracy: 0.9127
k49_shape = (28, 28, 1)
num_classes = 49

justin_base_model = Sequential()
justin_base_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=k49_shape))
justin_base_model.add(Conv2D(64, (3, 3), activation='relu', input_shape=k49_shape))
justin_base_model.add(Dropout(0.25))
justin_base_model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding="valid"))
justin_base_model.add(Conv2D(48, (3, 3), activation='relu', input_shape=k49_shape))
justin_base_model.add(Flatten())
justin_base_model.add(Dense(num_classes, activation='relu'))
justin_base_model.add(Dense(num_classes))

justin_base_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer="adam",
              metrics=['accuracy'])

justin_base_model.summary()


# Helper to load the base model and prepare it for transfer learning
def load_base_model():
    bm = justin_base_model
    bm.pop()
    bm.pop()                    # Cut off layers that flatten data dimensionality
    bm.pop()
    bm.trainable = False        # Base model MUST NOT be trainable
    return bm


# ==============================================================================
# NO OPTUNA: Transfer learning top model with unoptimized hyperparameters
# val_accuracy: 0.9241
kshape = (28, 28, 1)

final_base = load_base_model()

justin_basic_tl_model = Sequential()
justin_basic_tl_model.add(tl_base)                              # Base..
justin_basic_tl_model.add(Conv2D(48, (3, 3), activation='sigmoid', input_shape=kshape))
justin_basic_tl_model.add(AveragePooling2D(pool_size=(2, 2)))
justin_basic_tl_model.add(Dropout(0.2))
justin_basic_tl_model.add(Conv2D(32, (3, 3), activation='sigmoid'))
justin_basic_tl_model.add(Flatten())
justin_basic_tl_model.add(Dense(150, activation='relu'))
justin_basic_tl_model.add(Dense(150))

justin_basic_tl_model.summary()


# ==============================================================================
# OPTUNA: Transfer learning top model for the optimal optuna hyperparameters
# val_accuracy: 0.9368
kshape = (28, 28, 1)

final_base = load_base_model()

justin_optuna_tl_model = Sequential()
justin_optuna_tl_model.add(final_base)
justin_optuna_tl_model.add(Conv2D(63, (3, 3), activation='sigmoid', input_shape=kshape))
justin_optuna_tl_model.add(AveragePooling2D((2, 2)))
justin_optuna_tl_model.add(Dropout(0.2))
justin_optuna_tl_model.add(Conv2D(36, (3, 3), activation='sigmoid', input_shape=kshape))
justin_optuna_tl_model.add(Flatten())
justin_optuna_tl_model.add(Dense(152, activation='tanh'))
justin_optuna_tl_model.add(Dense(150))

justin_optuna_tl_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(), 'accuracy'])

justin_optuna_tl_model.summary()
