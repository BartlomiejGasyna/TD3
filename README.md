# TD3
TD3 - Deep reinforcement learning algorithm implementation in TensorFlow2

Projekt przedstawia implementacje algorytmu uczenia ze wzmocnieniem TD3 (RL - reinforcement learning) przy pomocy środowiska **gym**. Jest on rozwinięciem algorytmu DDPG i tak jak on został przeznaczony do zastosować w środowiskach ciągłych.

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

Działanie algorytmu TD3 przedstawia poniższy pseudokod:

![image](https://github.com/BartlomiejGasyna/TD3/assets/65308689/8be96ba9-ab49-45b4-994b-2cdbc61c5cbb)

Testy algorytmu zostały przeprowadzone w środowisku **BipedalWalker-v3** oraz **Pendulum-v0**. Procesy uczenia zostały przedstawione na poniższych przebiegach:
