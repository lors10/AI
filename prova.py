import re


class TuaClasse:
    def __init__(self, stringa_input):
        pttn = re.compile("\s*([\d]+)\s*")
        risultati = pttn.findall(stringa_input)

        if risultati and len(risultati) == 16:
            self.state = [[int(numero) for numero in risultati]]
        else:
            raise ValueError("Formato improprio per il 15-puzzle")


# Esempio di utilizzo
stringa_di_input = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0"
istanza = TuaClasse(stringa_di_input)
print(istanza.state)
