def info_o_sieci(potok, krok):
    print('Liczba warstw: ', potok.named_steps[krok].n_layers_)
    print('Liczba neuronów w warstwie wejściowej: ', potok.named_steps[krok].n_features_in_)
    print('Liczba neuronów w warstwach ukrytych: ', potok.named_steps[krok].hidden_layer_sizes)
    print('Funkcja aktywacji w warstwach ukrytych : ', potok.named_steps[krok].activation)
    print('Liczba neuronów w warstwie wyjściowej: ', potok.named_steps[krok].n_outputs_)
    print('Funkcja aktywacji w warstwie wyjściowej : ', potok.named_steps[krok].out_activation_)