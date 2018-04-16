import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Model

'''
m is the compiled mode
x_train and y_train is the data
num_epochs is the number of epochs
val_split is the validation split
bat_size is the batch_size
file_path is where your model will be saved
    example: "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    this will save your model with the file name including the number of epochs completed and the validation accuracy
    Warning: variable file_path will result in creating a new file everytime the model improves over the last one saved
'''
def saveModel(m, x_train, y_train, num_epochs, val_split, bat_size, filepath):
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    m.fit(x_train, y_train, 
          validation_split=val_split,
          batch_size=bat_size,
          epochs=num_epochs,
          callbacks=callbacks_list,
          verbose=1)
'''
m is the compiled model
x_train and y_train is the data
num_epochs is the number of epochs
val_split is the validation split
bat_size is the batch_size
file_path is where your model will be saved
    example: "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    this will save your model with the file name including the number of epochs completed and the validation accuracy
    Warning: variable file_path will result in creating a new file everytime the model improves over the last one saved
init_epoch is the last epoch # that was trained

Only use this one if you have loaded in a model


'''
def saveModel(m, x_train, y_train, num_epochs, val_split, bat_size, filepath, init_epoch=1):
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    m.fit(x_train, y_train, 
          validation_split=val_split,
          batch_size=bat_size,
          epochs=num_epochs+init_epoch+1,
          initial_epoch=init_epoch,
          callbacks=callbacks_list,
          verbose=1)

