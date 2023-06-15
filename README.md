# Twin Delayed DDPG (Deep Deterministic Policy Gradient)
TD3 - Deep reinforcement learning algorithm implementation in TensorFlow2

Projekt przedstawia implementacje algorytmu uczenia ze wzmocnieniem TD3 (RL - reinforcement learning) przy pomocy środowiska **gym**. Jest on rozwinięciem algorytmu DDPG i tak jak on został przeznaczony do zastosować w środowiskach ciągłych.

Testy algorytmu zostały przeprowadzone w środowisku **Pendulum-v0** oraz **BipedalWalker-v3**. Procesy uczenia zostały przedstawione na poniższych przebiegach:

<p align="center">
  <img src="https://github.com/BartlomiejGasyna/TD3/blob/master/results/pendulum_training.png" alt="Image 1" width="49%">
  <img src="https://github.com/BartlomiejGasyna/TD3/blob/master/results/bipedal_training.png" alt="Image 2" width="49%">
</p>


Kod został przygotowany na podstawie artykułu naukowego **Addressing Function Approximation Error in Actor-Critic Methods**, napisanego przez **Scott Fujimoto, Herke van Hoof, David Meger**. https://arxiv.org/pdf/1802.09477.pdf

Przedstawia on algorytm Q-learning w uczeniu ze wzmocnieniem, który służy do uczenia agenta podejmowania optymalnych decyzji w dynamicznym środowisku.

Pierwszym elementem jest zastosowanie koncepcji aktor-krytyk. Aktor reprezentuje polityke agenta, natomiast krytyk odpowiedzialny jest za ocene wartości akcji w danym stanie. Algorytm TD3 wykorzystuje głębokie sieci neuronowe do parametryzacji aktora oraz krytyka.

Ważnym aspektem jest wykorzystanie dwóch krytyków w celu redukcji szumów i stabilizacji uczenia. Każdy z nich działa niezależnie ale do procesu wybiera się mniejszą wartość zwracaną przez krytyków.

W algortymie TD3 zastosowane zostały tzw. target networks. Są to bliźniacze sieci (aktora i krytyka) które służą do stabilizacji uczenia w algorytmach RL opartych na głębokich sieciach neuronowych.
Służą one do oddzielenia ocenianych wartości Q od aktualizowanych wartości w procesie uczenia, co prowadzi do zwiększenia zbieżnosci algorytmów RL.
Sieci docelowe są aktualizowane wolniej niż sieci "aktualne", co pozwala na stabilizację procesu uczenia.

Innym istotnym elementem TD3 jest wykorzystanie odłożonej strategii aktualizacji (delayed policy updates).
Aktualizacje polityki (parametrów aktora) są opóźniane, aby zmniejszyć korelację między kolejnymi aktualizacjami i zapobiec destabilizacji procesu uczenia.

Algorytm TD3 wykorzystuje techniki gradientu prostego (stochastycznego) do aktualizacji wag sieci aktora i krytyków.
Poprzez iteracyjną optymalizację sieci neuronowych, TD3 stara się znaleźć optymalne parametry polityki agenta, które minimalizują funkcję kosztu i maksymalizują zbierane nagrody.

Mimo że DDPG czasem może osiągać doskonałe wyniki, często jest podatny na zmiany w hiperparametrach i inne rodzaje strojenia. Powszechnym błędem DDPG jest to, że nauczona funkcja Q zaczyna dramatycznie przeszacowywać wartości Q, co prowadzi do złamania polityki, ponieważ wykorzystuje błędy w funkcji Q. Twin Delayed DDPG (TD3) to algorytm, który rozwiązuje ten problem, wprowadzając trzy istotne usprawnienia:

**1** : Ograniczone podwójne uczenie Q. TD3 uczy dwóch funkcji Q zamiast jednej (stąd "twin") i używa mniejszej z dwóch wartości Q do tworzenia celów w funkcjach straty błędu Bellmana.

**2** : "Opóźnione" aktualizacje polityki. TD3 aktualizuje politykę (i sieci docelowe) rzadziej niż funkcję Q. W artykule zaleca się jedno aktualizowanie polityki dla dwóch aktualizacji funkcji Q.

**3** : Wygładzanie polityki docelowej. TD3 dodaje szum do docelowej akcji, aby utrudnić polityce wykorzystywanie błędów w funkcji Q przez wygładzanie Q podczas zmian w akcji.

Działanie algorytmu TD3 przedstawia poniższy pseudokod:
<p align="center">
<img src="https://github.com/BartlomiejGasyna/TD3/assets/65308689/8be96ba9-ab49-45b4-994b-2cdbc61c5cbb" width=40% height=40%  >
</p>


