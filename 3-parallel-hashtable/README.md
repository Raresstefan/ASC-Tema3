# Implementarea solutiei

Algoritmul de inserare este bazat pe o functie de calculare a hash-ului pentru o anumita cheie
si pe strategia de rezolvare a coliziunilor numita "linear probing".

Pentru redimensionarea tabelei se creaza o noua tabela cu dimensiunea dorita si se copiaza fiecare element din tabela veche
in cea noua. Aceasta abordare permite adaptarea dimensiunii tabelei hash la numarul de elemente si mentinerea unei
performante optime.

In ceea ce priveste gasirea unei chei in tabela hash, fiecare thread va cauta o anumita cheie dintre cele dorite
pana cand cheia corespunzatoare hash-ului calculat este gasita, iar valoarea acesteia se va stoca intr-un vector.

# Stocarea datelor

Tabela hash este stocata in VRAM. La initializarea obiectului GpuHashTable se aloca memorie pentru tabela
hash, alocandu-se memorie pe GPU si asigurand atat accesul de pe CPU, cat si de pe GPU. Tabela este stocata sub forma
unui vector de elemente de tip HashElement alocate in VRAM. Structura HashElement retine cheia si valoarea
corespunzatoare acesteia.

# Performante obtinute

In ceea ce priveste performanta inserarii, aceasta depinde de cat de plina este tabela. Atunci cand
tabela depaseste un anumit factor de umplere(LOAD_FACTOR_MAX), se redimensioneaza tabela pentru a evita
coliziunile excesive si pentru a mentine un timp de insertie cat mai bun. Avand in vedere ca redimensionarea
presupune realocarea memoriei si copierea elementelor in noua tabela redimensionata aceasta operatie poate fi costisitoare.
Totusi, redimensionarea are loc doar atunci cand tabela hash devine prea plina, astfel in medie performanta insertiei
ramane buna in majoritatea cazurilor.

In ceea ce priveste cautarea cheilor in tabela, neutilizandu-se operatii atomice, intrucat tabela nu este modificata
in urma acestei operatii, functia de cautare a cheilor se executa considerabil mai rapid decat cea de inserare in tabela.