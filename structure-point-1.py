# Definizione del modulo/classe Gioco
class Gioco:
    def __init__(self):
        self.stato_corrente = 0

    def transizione(self, azione):
        # Simulazione di una transizione di stato
        self.stato_corrente += azione

    def vicinato(self):
        # Restituzione di situazioni simili
        return [self.stato_corrente - 1, self.stato_corrente, self.stato_corrente + 1]

# Definizione del modulo/classe Euristiche
class Euristiche:
    @staticmethod
    def funzione_euristica(situazione):
        # Esempio di una funzione euristica semplice
        return situazione * 2

# Definizione del modulo/classe Agente che utilizza la Ricerca Euristica
class Agente_astar:
    def __init__(self, gioco, euristiche):
        self.gioco = gioco
        self.euristiche = euristiche

    def scegli_azione(self):
        situazione_corrente = self.gioco.stato_corrente
        azioni_possibili = self.gioco.vicinato()

        # Utilizza la funzione euristica per valutare le azioni possibili
        valutazioni = [self.euristiche.funzione_euristica(s) for s in azioni_possibili]

        # Scegli l'azione con la valutazione massima
        azione_migliore = azioni_possibili[valutazioni.index(max(valutazioni))]

        return azione_migliore

class Agente_bfs:
    def __init__(self, gioco, euristiche):
        self.gioco = gioco
        self.euristiche = euristiche

    def scegli_azione(self):
        situazione_corrente = self.gioco.stato_corrente
        azioni_possibili = self.gioco.vicinato()

        # Utilizza la funzione euristica per valutare le azioni possibili
        valutazioni = [self.euristiche.funzione_euristica(s) for s in azioni_possibili]

        # Scegli l'azione con la valutazione massima
        azione_migliore = azioni_possibili[valutazioni.index(max(valutazioni))]

        return azione_migliore

# Creazione delle istanze dei moduli/classi
gioco = Gioco()
euristiche = Euristiche()
agente = Agente(gioco, euristiche)

# Esempio di utilizzo
for _ in range(5):
    azione_scelta = agente.scegli_azione()
    gioco.transizione(azione_scelta)
    print(f"Stato corrente: {gioco.stato_corrente}, Azione scelta: {azione_scelta}")
