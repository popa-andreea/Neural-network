# Neural-network

train_perceptron(X, y, epcochs, learning_rate)
-
X si y reprezinta datele de antrenare, respectiv etichetele acestora 

epochs reprezinta numarul de epoci

learning_rate reprezinta rata de invatare
- initializam ponderile (W) cu un vector de 0-uri si bias-ul cu 0;
- pentru fiecare epoca, amestecam datele de antrenare;
- pentru fiecare exemplu x din datele de antrenare, calculam predictia (y_hat) dupa formula: x * W + bias;
- calculam eroare (loss) pentru exemplul x dupa formula: (y_hat-y)^2 / 2, unde y reprezinta eticheta lui x;
- actualizam ponderile si bias-ul;
- calculam acuratetea;
- afisam grafic dreapta de decizie: [x, y] * [W[0], W[1]] + b = 0.

train_neural_network(X, y, epochs, learning_rate, no_hidden, no_out)
-
X si y reprezinta datele de antrenare, respectiv etichetele acestora

epochs reprezinta numarul de epoci

learning_rate reprezinta rata de invatare

no_hidden reprezinta numarul de neuroni al stratului ascuns

no_out reprezinta numarul de neuroni al predictiei (output-ului)
- initializam aleator ponderile si bias-ul cu valori mici aproape de 0;
- pentru fiecare epoca, amestecam datele;
- calculam cu ajurotul metodei forward pentru fiecare strat valoarea lui z (z = X * W + bias) si valoarea lui a (a = f(z), unde f reprezinta functia de activare), iar valoarea finala a lui a reprezinta predictia retelei (output-ul);
- calculam valorea functiei de eroare si acuratetea;
- calculam cu ajutorul metodei backward derivata functiei de eroare pe directiile ponderilor, respectiv a fiecarui bias folosind regula de inlantuire: 
X, W1, b1 -> z1 -> f(z1) = a1
a1, W2, b2 -> z2 -> f(z2) = a2 (output)
a2, y -> loss;
- actualizam proportional poderile si bias-urile cu negativul mediei derivatelor din batch;
- afisam grafic functia de decizie.
