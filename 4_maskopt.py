from layers.utils import fft_conv2d, fft_conv2d_adjoint, optical_conv_layer

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from glob import glob
from datetime import datetime
import abc
from pprint import pprint
import tensorflow as tf2

gpus = tf2.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf2.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf2.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

CONFIG = type('', (), {})() # object for params
CONFIG.image_size = 192
CONFIG.scale = 4
CONFIG.phase_mask_shape = np.array((CONFIG.scale * CONFIG.image_size, CONFIG.scale * CONFIG.image_size))
CONFIG.wavelength = 550e-9
CONFIG.refraction_index = 1.509
CONFIG.num_steps = 200001
CONFIG.starter_learning_rate = 0.0001
CONFIG.batch_size = 1
CONFIG.psf_file_path = 'assets/quickdraw16_tiledpsf.npy'
CONFIG.optimizer = 'ADAM'
CONFIG.opt_params = {'momentum': 0.5, 'use_nesterov': True}
CONFIG.decay_policy = 'polynomial'
CONFIG.decay_policy_params = {'decay_steps': CONFIG.num_steps, 'end_learning_rate': 1e-10}
CONFIG.num_steps_until_save = 500
CONFIG.num_steps_until_summary = 50

now = datetime.now()
CONFIG.run_id = 'quickdraw16_tiledpsf_4x/' + now.strftime('%Y%m%d-%H%M%S') + '/'
CONFIG.log_dir = os.path.join('checkpoints/maskopt/', CONFIG.run_id)

print()
pprint(CONFIG.__dict__)
print()

if tf.io.gfile.exists(CONFIG.log_dir):
    tf.io.gfile.rmtree(CONFIG.log_dir)
tf.io.gfile.makedirs(CONFIG.log_dir)

class Model(abc.ABC):
    """Generic tensorflow model class.
    """
    def __init__(self, name, ckpt_path=None):
        sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self.sess = tf.compat.v1.Session(config=sess_config)
        self.name = name
        self.ckpt_path = ckpt_path

    @abc.abstractmethod
    def _build_graph(self, x_train, **kwargs):
        """Builds the model, given x_train as input.

        Args:
            x_train: The dequeued training example
            **kwargs: Model parameters that can later be passed to the "fit" function

        Returns:
            model_output: The output of the model
        """

    @abc.abstractmethod
    def _get_data_loss(self,
                      model_output,
                      ground_truth):
        """Computes the data loss (not regularization loss) of the model.

        !!For consistency of weighing of regularization loss vs. data loss,
        normalize loss by batch size!!

        Args:
            model_output: Output of self._build_graph
            ground_truth: respective ground truth

        Returns:
            data_loss: Scalar data loss of the model.         """

    def _get_reg_loss(self):
        reg_loss = tf.reduce_sum(input_tensor=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        return reg_loss


    def infer(self, model_inputs, model_params={}, get_loss=False, gt=None):
        """Does inference at test time.
        """
        x_test, y_test = self._get_inference_queue()

        # Set up the training graph
        with tf.compat.v1.variable_scope('model'):
            model_output_graph = self._build_graph(x_test, **model_params)

            if get_loss:
                data_loss_graph = self._get_data_loss(model_output_graph, y_test)

        # Create a saver
        self.saver = tf.compat.v1.train.Saver()

        if self.ckpt_path is not None:
            self.saver.restore(self.sess,self.ckpt_path)
        else:
            print("Warning: No checkpoint path given. Inference happens with random weights")

        # Init op
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        print("Starting Queues")
        coord = tf.train.Coordinator()
        enqueue_threads = tf.compat.v1.train.start_queue_runners(coord=coord, sess=self.sess)

        model_outputs = []
        try:
            while True:
                model_output= self.sess.run(model_output_grpah)
                model_outputs.append(model_output)

                if coord.should_stop():
                   break

        except Exception as e:
            print("Interrupted due to exception")
            print(e)
            coord.request_stop()
        finally:
            coord.request_stop()
            coord.join(enqueue_threads)

        return model_outputs


    @abc.abstractmethod
    def _get_training_queue(self, batch_size):
        """Builds the queues for training data.

        Use tensorflow's readers, decoders and tf.train.batch to build the dataset.

        Args:
            batch_size:

        Returns:
            x_train: the dequeued model input
            y_train: the dequeued ground truth

        Sketch of minimum example:
            def _get_training_queue(self, batch_size):
                file_list = tf.matching_files('./test_imgs/*.png')
                filename_queue = tf.train.string_input_producer(file_list)

                image_reader = tf.WholeFileReader()
                _, image_file = image_reader.read(filename_queue)
                image = tf.image.decode_png(image_file,
                                            channels=1,
                                            dtype=tf.uint8)
                image = tf.cast(image, tf.float32)
                image /= 255.0

                image_batch = tf.train.batch(image,
                                             shapes=[512,512,1],
                                             batch_size=batch_size)
                return image_batch
        """


    def _get_validation_queue(self):
        """

        Returns:

        """

    def fit(self,
            model_params, # Dictionary of model parameters
            opt_type, # Type of optimization algorithm
            opt_params, # Parameters of optimization algorithm
            batch_size,
            starter_learning_rate,
            logdir,
            num_steps,
            num_steps_until_save,
            num_steps_until_summary,
            adadelta_learning_rate=0.1,
            decay_type=None, # Type of decay
            decay_params=None, # Decay parameters
            ):
        """Trains the model.
        """
        x_train, y_train = self._get_training_queue(batch_size)

        print("\n\n")
        print(40*"*")
        print("Saving model and summaries to %s"%logdir)
        print("Optimization parameters:")
        print(opt_type)
        print(opt_params)
        print("Starter learning rate is %f"%starter_learning_rate)
        print(40*"*")
        print("\n\n")

        # Set up the training graph
        with tf.compat.v1.variable_scope('model'):
            model_output_train = self._build_graph(x_train, **model_params)
            data_loss_graph = self._get_data_loss(model_output_train, y_train)
            reg_loss_graph = self._get_reg_loss()
            total_loss_graph = tf.add(reg_loss_graph,
                                      data_loss_graph)

        if decay_type is not None:
            global_step = tf.Variable(0, trainable=False)

            if decay_type == 'exponential':
                learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,
                                                           global_step,
                                                           **decay_params)
            elif decay_type == 'polynomial':
                learning_rate = tf.compat.v1.train.polynomial_decay(starter_learning_rate,
                                                           global_step,
                                                           **decay_params)
        else:
            learning_rate = starter_learning_rate

        if opt_type == 'ADAM':
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate) #,
                                               #**opt_params)
        elif opt_type == 'sgd_with_momentum':
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate,
                                               **opt_params)
        elif opt_type == 'adadelta':
            optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=adadelta_learning_rate, rho=.9)
                                               #**opt_params)

        if decay_type is not None:
            train_step = optimizer.minimize(total_loss_graph, global_step=global_step)
        else:
            train_step = optimizer.minimize(total_loss_graph)

        # Attach summaries to some of the training parameters
        tf.compat.v1.summary.scalar('data_loss', data_loss_graph)
        tf.compat.v1.summary.scalar('reg_loss', reg_loss_graph)
        tf.compat.v1.summary.scalar('total_loss', total_loss_graph)
        tf.compat.v1.summary.scalar('learning_rate', learning_rate)

        # Create a saver
        self.saver = tf.compat.v1.train.Saver(keep_checkpoint_every_n_hours=2,
                                    max_to_keep=3)

        # Get all summaries
        summaries_merged = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter(logdir, self.sess.graph, flush_secs=60)

        if self.ckpt_path is not None:
            self.saver.restore(self.sess,self.ckpt_path)

        # Init op
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        # Train the model
        print("Starting Queues")
        coord = tf.train.Coordinator()
        enqueue_threads = tf.compat.v1.train.start_queue_runners(coord=coord, sess=self.sess)

        print("Beginning the training")
        try:
            for step in range(num_steps):
                _, total_loss, reg_loss, data_loss = self.sess.run([train_step,
                                                                    total_loss_graph,
                                                                    reg_loss_graph,
                                                                    data_loss_graph])
                if not step % 20:
                    print("\r Step %d  total_loss %0.8f   reg_loss %0.8f   data_loss %0.8f"%\
                        (step, total_loss, reg_loss, data_loss), end="")

                if coord.should_stop():
                   break

                if not step % num_steps_until_save and step:
                    print("\r Saving model...", end="")
                    save_path = os.path.join(logdir, self.name+'.ckpt')
                    self.saver.save(self.sess, save_path, global_step=step)

                if not step % num_steps_until_summary:
                    print("\r Writing summaries...", end="")
                    summary = self.sess.run(summaries_merged)
                    summary_writer.add_summary(summary, step)
        except Exception as e:
            print("Training interrupted due to exception")
            print(e)
            coord.request_stop()
        finally:
            coord.request_stop()
            coord.join(enqueue_threads)

class PhaseMaskModel(Model):
    def __init__(self, 
        psf_file_path,
        image_size, 
        phase_mask_shape,
        wavelength,
        refraction_index,  
        ckpt_path=None):

        self.image_size = image_size
        self.phase_mask_shape = phase_mask_shape
        self.wavelength = wavelength
        self.refraction_index = refraction_index
        self.psf_file_path = psf_file_path

        super(PhaseMaskModel, self).__init__(name='PhaseMask_ONN', ckpt_path=ckpt_path)

    def _build_graph(self, x_train):
        sensordims = (self.image_size, self.image_size)
        
        # input_img = x_train/tf.reduce_sum(input_tensor=x_train) 
        # why energy is calculated for batch, not per image? - because batch was 1 always
        input_img = x_train/tf.reduce_sum(input_tensor=x_train, axis=[1,2,3], keep_dims=True) # conservation of energy

        input_img = tf.image.resize(
            input_img, 
            size = self.phase_mask_shape, 
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        output_fullres = optical_conv_layer(
            input_img, 
            refraction_index = self.refraction_index, 
            wavelength = self.wavelength,
            name = 'maskopt'
        )

        output_img = tf.image.resize(
            output_fullres, 
            size = sensordims, 
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        
        return output_img

    def _get_data_loss(self, model_output, ground_truth):
        model_output = tf.cast(model_output, tf.float32)
        ground_truth = tf.cast(ground_truth, tf.float32)
        
        # model_output /= tf.reduce_max(model_output)
        # ground_truth /= tf.reduce_sum(input_tensor=ground_truth, axis=[1,2,3], keep_dims=True)
        # ground truth energy conservation is done two times
        # with tf.compat.v1.name_scope('data_loss'):
        #     optics.attach_img('model_output', model_output)
        #     optics.attach_img('ground_truth', ground_truth)
        loss = tf.reduce_mean(input_tensor = tf.abs(model_output - ground_truth))
        return loss

    def _get_training_queue(self, batch_size, num_threads=4):
        image_size = self.image_size
        
        file_list = tf.io.matching_files('assets/quickdraw16_192/im_*.png')
        filename_queue = tf.compat.v1.train.string_input_producer(file_list)
      
        image_reader = tf.compat.v1.WholeFileReader()

        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_png(
            image_file,
            channels = 1,
            dtype = tf.uint8
        )
        image = tf.cast(image, tf.float32) # Shape [height, width, 1]
        image = tf.expand_dims(image, 0)
        image /= 255.

        # this method is used for padding with zeros or cropping
        image = tf.image.extract_glimpse(
            image, 
            size = [image_size, image_size], 
            offsets = [[0, 0]], 
            centered = True, 
            normalized = False
        )
        
        psf = np.load(self.psf_file_path)
        psf = tf.convert_to_tensor(psf, tf.float32)
        psf /= tf.reduce_sum(psf) # conservation of energy
        
        psf = tf.expand_dims(tf.expand_dims(tf.squeeze(psf), -1), -1)
        
        # convolved_image = fft_conv2d_adjoint(image, psf) # the result is rotated 180 deg
        convolved_image = fft_conv2d(image, psf)

        convolved_image = tf.abs(convolved_image)
        convolved_image /= tf.reduce_sum(convolved_image, axis=[1,2,3], keep_dims=True) # conservation of energy

        image = tf.squeeze(image, axis=0)
        convolved_image = tf.squeeze(convolved_image, axis=0)

        image_batch, convolved_img_batch = tf.compat.v1.train.batch(
            [image, convolved_image],
            shapes = [[image_size,image_size,1], [image_size,image_size,1]],
            batch_size = batch_size,
            num_threads = 4,
            capacity = 4*batch_size
        )

        return image_batch, convolved_img_batch


tf.compat.v1.reset_default_graph() # why ?

phasemask = PhaseMaskModel(
    psf_file_path = CONFIG.psf_file_path, 
    image_size = CONFIG.image_size, 
    phase_mask_shape = CONFIG.phase_mask_shape, 
    wavelength = CONFIG.wavelength, 
    refraction_index = CONFIG.refraction_index, 
    ckpt_path = None
)

phasemask.fit(
    model_params = {},
    opt_type = CONFIG.optimizer,
    opt_params = CONFIG.opt_params,
    decay_type = CONFIG.decay_policy,
    decay_params = CONFIG.decay_policy_params,
    batch_size = CONFIG.batch_size,
    starter_learning_rate = CONFIG.starter_learning_rate,
    num_steps_until_save = CONFIG.num_steps_until_save,
    num_steps_until_summary = CONFIG.num_steps_until_summary,
    logdir = CONFIG.log_dir,
    num_steps = CONFIG.num_steps
)

print("CONFIG:")
pprint(CONFIG.__dict__)
print()