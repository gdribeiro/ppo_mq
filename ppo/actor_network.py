

class ActorNet:

    
    def __init__(self):
      actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.observation_tensor_spec,
            self.action_tensor_spec,
            fc_layer_params=self.actor_fc_layers,
            activation_fn=tf.keras.activations.tanh,
            kernel_initializer=tf.keras.initializers.Orthogonal(),
            seed_stream_class=tfp.util.SeedStream
        )
       