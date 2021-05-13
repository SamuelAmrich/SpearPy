# SpearPy
Program na hľadanie ostrých Peak-ov v silne zašumených dátach.
Ako vyzerá kompletná stránka front-endu je možné nájsť na: 
[Dash.pdf](https://github.com/SamuelAmrich/SpearPy/files/6453927/Dash.pdf)

Prvá časť je linka kde je: 
  - Výber dataset-u z dropdown menu (Funguje)
  - Ručné navolenie cesty k dátam (Nefunguje, všetky dáta sa dávajú do priečinka data v programovom priečinku)
  - Voľba rozlišovacej presnosti grafov (Funguje)
  - Tlačidlo na spustenie programu (Irelevantné, program sa spúšťa automaticky)
  -  Voľba uloźených nastavení (zatiaľ nefunguje)
  -  Cesta k uloženým dátam (zatiaľ nefunguje)
![image](https://user-images.githubusercontent.com/55489761/118136746-bb4a2700-b404-11eb-858d-12efeb69bc51.png)


Druhá časť je graf zobrazujúci originálne dáta:
  - Graf je interaktívny, dá sa posúvať, približovať a pod.
![image](https://user-images.githubusercontent.com/55489761/118136952-f0ef1000-b404-11eb-8626-b4b0030037ad.png)

Tretia časť je linka pre FFT filter:
  - Máme na výber 2 vstupy. Pričom na grafe sa nám zobrazuje frekvencie ktoré sa v dátach vyskytujú, a zároveň krivka ktorá určuje filter (Gauss krivka)
  - - "a" určuje miesto posunutia maxima filtrovacej krivky.
  - - "σ" určuje šírku filtra
![image](https://user-images.githubusercontent.com/55489761/118137942-0fa1d680-b406-11eb-9443-c691f2bfbe77.png)

Štvrtá časť je linka pre vyhladzovanie dát pomocou Savitsky-Golay filter:
  - Máme na výber 2 vstupy, pričom na grafe je vidieť výsledná krivka po prechode FFt filtrom aj S-G filtrom.
  - -"win" je množstvo dát v okolí každého bodu ktoré má filter brať do úvahy
  - -"pol" je úroveň polynómu ktorým filter vyhladzuje dáta
![image](https://user-images.githubusercontent.com/55489761/118138570-c7cf7f00-b406-11eb-94da-bb80c8ccbbbd.png)

Piata časť je samotný vyhľadávač:
  - Máme na výber 6 vstupov, pričom na grafe je vidieť výsledná krivka aj s označenými bodmi ktoré sa našli.
  - -"Tr+" označuje o koľko sa musí hodnota maximá líši´t od predchádzajúceho bodu
  - -"Tr-" obdobne ako "Tr+" ale pre záporné hodnoty
  - -"dis" Hovorí aká je minimálkna vzdialenosť medzi dvoma po sebe idúcimi vrcholmi
  - -"pro" Hovorí ako veľmi výrazný mui byť vrchol zo štatistického pohľadu
  - -"wid" Hovorí aká je minimálna šírka vrcholu
  - -"hei" Hovorí aká je minimálna výška vrcholu
![image](https://user-images.githubusercontent.com/55489761/118139967-3234ef00-b408-11eb-8746-00e03635dc3c.png)

Šiesta časť je ukladacia:
  - Môžeme si navoliť do akých formátov sa uložia výsledky a je tam tlačidlo ktoré dáta ukladá (v súčasnosti sa uloźia ysledky po každej zmene)
  - -".TXT" Uloží polohy vrcholov ako textový súbor (zatial jedina moznost)
  - -".CSV" Uloží polohy vrcholov ako csv súbor (este nedotiahnute)
  - -".PNG" Ulozí výsledný graf ako png obrázok
  - -".PDF" Uloží výsledený graf ako pdf súbor
  - -"QLS" Uloží nastavenie všetkých parametrov ako vlastný typ súbory a ktorý si potom vie naloadovať pri ďalšom spustení (ešte vyžaduje dokočiť) 
