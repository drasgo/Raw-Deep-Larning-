# Raw-Deep-Learning
Set of raw implementations of Deep Learning techniques

The framework can be used by creating the structure:

     import RDL
    
     network = RDL.NeuralNetworks.MLP() #optional parameter verbosity: RDL.NeuralNetwork.Verbosity.RELEASE/DEBUG 
    
     net.add_normalization(False) # Optional: default True
    
     net.add_normalization(False) # Optional: default False
    
     net.add_input_layer(
        layer_name="input", 
        input_nodes=2,
    )
    
    net.add_layer(
        layer_name="h1",
        input_layer="input",
        nodes=3,
        activation_function=ActivationFunctions.SIGMOID, 
        weight_initialization=WeightInitializations.COSTUM, # Optional default ones
        custum_weigths=numpy.array([[1, -1], [1, -1], [1, -1]]), #Optional, only for WeightInitializations.COSTUM,
        bias_initialization: WeightInitializations = WeightInitializations.ZERO, # Optional default zeros
    )
    
    net.add_output_layer(
        layer_name="output",
        input_layer="h1",
        nodes=2,
        activation_function=ActivationFunctions.SOFTMAX, # Optional
        weight_initialization=WeightInitializations.COSTUM, # Optional
        custum_weigths=numpy.array([[1, -1, -1], [1, -1, -1]]),
        bias_initialization: WeightInitializations = WeightInitializations.ZERO, # Optional default zeros

    )
    net.add_loss_function(LossFunctions.CROSS_ENTROPY)
    net.add_learning_rate(0.001)
    
    net.train(
        input_data=input_train,
        target_data=target_train,
        validation_input=input_validation,#Optional
        validation_target=target_validation, #Optional
        epochs=1, #Optional
        batch_size=1,#Optional
    )
    net.test(
        test_input=input,
        test_target=target)

