import numpy as np
import tensorflow as tf

DEFAULT_MARGIN = 15


class _DistanceLayer(tf.keras.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


class _SiameseModel(tf.keras.Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    
    Where A,P,N are the anchor, positive, and negative inputs, respecively,
    and f is a function that takes in an input and outputs its encoding.
    """

    def __init__(self, siamese_network, margin=DEFAULT_MARGIN):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        ) 

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


def train_model(data, margin=DEFAULT_MARGIN):
    num_features = len(data[0])
    target_shape = (num_features,)

    def generate_triplets():
        anchors, positives = np.vsplit(data, 2)
        negatives = anchors + positives

        positive_distances = np.linalg.norm(anchors - positives,axis=1)
        negative_distances = np.linalg.norm(anchors - negatives,axis=1)

        positive_less_than_negative = positive_distances < negative_distances
        negative_less_than_positive_plus_margin = negative_distances < positive_distances + margin
        mask =  positive_less_than_negative & negative_less_than_positive_plus_margin

        anchors, positives, negatives = anchors[mask], positives[mask], negatives[mask]

        anchor_dataset = tf.data.Dataset.from_tensor_slices(anchors)
        positive_dataset = tf.data.Dataset.from_tensor_slices(positives)
        

        rng = np.random.RandomState(seed=42)
        rng.shuffle(anchors)
        rng.shuffle(positives)

        np.random.RandomState(seed=32).shuffle(negatives)

        negative_dataset = tf.data.Dataset.from_tensor_slices(negatives)
        negative_dataset = negative_dataset.shuffle(buffer_size=4096)

        dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
        dataset = dataset.shuffle(buffer_size=1024)

        return dataset
    
    def split_train_test(dataset, data_size):
        train_dataset = dataset.take(round(data_size * 0.8))
        val_dataset = dataset.skip(round(data_size * 0.8))

        train_dataset = train_dataset.batch(32, drop_remainder=False)
        train_dataset = train_dataset.prefetch(8)

        val_dataset = val_dataset.batch(32, drop_remainder=False)
        val_dataset = val_dataset.prefetch(8)

        return train_dataset, val_dataset

    def embedding_model():
        embedding = tf.keras.models.Sequential()
        embedding.add(tf.keras.layers.Dense(2*num_features//3 + num_features, activation="relu", input_shape=target_shape))
        embedding.add(tf.keras.layers.BatchNormalization())
        embedding.add(tf.keras.layers.Dense(2*num_features//3 + num_features, activation="relu"))
        embedding.add(tf.keras.layers.BatchNormalization())
        embedding.add(tf.keras.layers.Dense(num_features))
        
        return embedding
    
    def siamese_network(embedding):
        anchor_input = tf.keras.layers.Input(name="anchor", shape=target_shape)
        positive_input = tf.keras.layers.Input(name="positive", shape=target_shape)
        negative_input = tf.keras.layers.Input(name="negative", shape=target_shape)

        distances = _DistanceLayer()(
            embedding(anchor_input),
            embedding(positive_input),
            embedding(negative_input)
        )

        siamese_network = tf.keras.Model(
            inputs=(anchor_input, positive_input, negative_input), outputs=distances
        )
        
        return siamese_network

    triplets = generate_triplets()
    train_dataset,val_dataset = split_train_test(triplets, len(data) // 2)
    embedding = embedding_model()
    siamese_network = siamese_network(embedding)
    siamese_model = _SiameseModel(siamese_network)
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), weighted_metrics=[tf.keras.metrics.Mean()])
    siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

    return embedding, siamese_model

