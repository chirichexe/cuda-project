// 
//
// Possiamo associare ciascun thread della griglia ai dati che i thread devono processare
// Si utilizzano le coordinate, ed esiste la "global index" (ix, iy, iz, grazie a quest'ultima)
// la dimensione del thread è identificata rispetto alla gerarchia
//
// Le operazioni su matrici sono molto veloci
// Vengono memorizzate in modo *lineare* nella memoria globale, utilizzando approcci *row-major*
// ovvero riga per riga. Per accedere agli elementi in memoria si usa la formula:
// 
// idx = i * width + j 
// - i = numero di righe che precedono quella che ci interessa, 
// - width = righe della matrice j 
// - j = posizione dell'elemento nella riga che ci interessa
// 
// Obiettivo: realizzare somma di matrici parallela in CUDA. 
// I passi per fare il kernel:
// 1. Prepara lato host le matrici che interessano
// 2. Utilizza le API per trasferirle sulla GPU
// 3. Definisci la dimensione dei blocchi: sempre un multiplo di 32
// 4. Una volta fatto, determina dimensione 
// devo scegliere un numero di thread quantomeno pari agli elementi da processare
