from keras import callbacks as k_callbacks
import json
from .data_utils.data_loader import image_segmentation_generator , verify_segmentation_dataset, get_pairs_from_paths
from .models import model_from_name
import os
import six

def find_latest_checkpoint( checkpoints_path ):
    ep = 0
    r = None
    while True:
        if os.path.isfile( checkpoints_path + "." + str( ep )  ):
            r = checkpoints_path + "." + str( ep ) 
        else:
            return r 

        ep += 1




def train( model  , 
        train_images  , 
        train_annotations , 
        input_height=None , 
        input_width=None , 
        n_classes=None,
        verify_dataset=True,
        checkpoints_path=None , 
        epochs = 5,
        batch_size = 2,
        validate=False , 
        val_images=None , 
        val_annotations=None ,
        val_batch_size=2 , 
        auto_resume_checkpoint=False ,
        load_weights=None ,
        steps_per_epoch=None,
        optimizer_name='adadelta',
        logging = False,
    ):


    if  isinstance(model, six.string_types) : # check if user gives model name insteead of the model object
        # create the model from the name
        assert ( not n_classes is None ) , "Please provide the n_classes"
        if (not input_height is None ) and ( not input_width is None):
            model = model_from_name[ model ](  n_classes , input_height=input_height , input_width=input_width )
        else:
            model = model_from_name[ model ](  n_classes )

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width


    if validate:
        assert not (  val_images is None ) 
        assert not (  val_annotations is None ) 

    if not optimizer_name is None:
        model.compile(loss='categorical_crossentropy',
            optimizer= optimizer_name ,
            metrics=['accuracy'])

    if not checkpoints_path is None:
        open( checkpoints_path+"_config.json" , "w" ).write( json.dumps( {
            "model_class" : model.model_name ,
            "n_classes" : n_classes ,
            "input_height" : input_height ,
            "input_width" : input_width ,
            "output_height" : output_height ,
            "output_width" : output_width 
        }))

    if ( not (load_weights is None )) and  len( load_weights ) > 0:
        print("Loading weights from " , load_weights )
        model.load_weights(load_weights)

    if auto_resume_checkpoint and ( not checkpoints_path is None ):
      latest_checkpoint = find_latest_checkpoint( checkpoints_path )
      if not latest_checkpoint is None:
            print("Loading the weights from latest checkpoint "  ,latest_checkpoint )
            model.load_weights( latest_checkpoint )
            start_epoch = int(os.path.splitext(latest_checkpoint)[-1][1:])+1 #extract number from checkpoint name
    else:
        start_epoch = 0


    if verify_dataset:
        print("Verifying train dataset")
        verify_segmentation_dataset( train_images , train_annotations , n_classes )
        if validate:
            print("Verifying val dataset")
            verify_segmentation_dataset( val_images , val_annotations , n_classes )
        

    train_gen = image_segmentation_generator( train_images , train_annotations ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   )
    if steps_per_epoch is None:
        steps_per_epoch = len(get_pairs_from_paths(train_images, train_annotations))
        print('steps per epoch', steps_per_epoch)

    if validate:
        val_gen  = image_segmentation_generator( val_images , val_annotations ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
        validation_steps = len(get_pairs_from_paths(val_images, val_annotations))
    else:
        val_gen = None

    if logging:
       tbCallBack = k_callbacks.TensorBoard(histogram_freq=0,log_dir=checkpoints_path)
       callbacks = [tbCallBack]
    else:
       callbacks = None
       
    model.fit_generator(train_gen, steps_per_epoch, validation_data = val_gen, validation_steps = validation_steps, epochs=epochs ,callbacks=callbacks ) #temporary fix, breaks the start_epoch functionality
    model.save_weights( checkpoints_path + "." + str( epochs ) )



