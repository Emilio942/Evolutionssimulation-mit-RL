import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import multiprocessing as mp
from collections import defaultdict
import time

class Umwelt:
    """Simulationsumgebung mit abiotischen Faktoren."""
    
    def __init__(self, breite=100, hoehe=100, nahrung_wachstumsrate=0.01):
        self.breite = breite
        self.hoehe = hoehe
        self.nahrungskarte = np.zeros((breite, hoehe))
        self.nahrung_wachstumsrate = nahrung_wachstumsrate
        self.jahreszeit = 0  # 0=Frühling, 1=Sommer, 2=Herbst, 3=Winter
        self.jahreszeiten_zyklus = 100  # Zeitschritte pro Jahreszeit
        self.aktuelle_zeit = 0
        
        # Terrain erstellen (Berge, Wasser, etc.)
        self.terrain = self._erstelle_terrain()
        # Initiale Nahrungsverteilung
        self._nahrung_verteilen()
    
    def _erstelle_terrain(self):
        """Erstellt eine zufällige Terrainstruktur mit Bergen, Wasser und Ebenen."""
        # Perlin-Noise-ähnlicher Ansatz für natürliches Terrain
        terrain = np.zeros((self.breite, self.hoehe))
        for i in range(5):  # Mehrere Frequenzschichten
            freq = 2**i
            amp = 0.5**i
            phase = np.random.randint(0, 100)
            for x in range(self.breite):
                for y in range(self.hoehe):
                    terrain[x, y] += amp * np.sin((x/freq + phase) * (y/freq + phase))
        
        # Normalisieren auf Werte zwischen 0 und 1
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
        return terrain
    
    def _nahrung_verteilen(self):
        """Verteilt Nahrung basierend auf Terrain und Jahreszeit."""
        # Nahrung wächst besser in Ebenen (terrain ~ 0.5) als in Bergen oder Wasser
        grundverteilung = 1.0 - 2.0 * np.abs(self.terrain - 0.5)
        
        # Jahreszeitlicher Einfluss
        if self.jahreszeit == 0:  # Frühling
            faktor = 1.2
        elif self.jahreszeit == 1:  # Sommer
            faktor = 1.5
        elif self.jahreszeit == 2:  # Herbst
            faktor = 0.8
        else:  # Winter
            faktor = 0.3
        
        # Aktualisiere Nahrungskarte
        self.nahrungskarte = np.clip(
            self.nahrungskarte + self.nahrung_wachstumsrate * grundverteilung * faktor,
            0, 1
        )
    
    def aktualisieren(self):
        """Aktualisiert die Umwelt für einen Zeitschritt."""
        self.aktuelle_zeit += 1
        
        # Jahreszeit aktualisieren
        if self.aktuelle_zeit % self.jahreszeiten_zyklus == 0:
            self.jahreszeit = (self.jahreszeit + 1) % 4
        
        # Nahrung aktualisieren
        self._nahrung_verteilen()
    
    def get_umwelt_zustand(self, position):
        """Liefert Umweltdaten für eine bestimmte Position."""
        x, y = int(position[0]), int(position[1])
        x = np.clip(x, 0, self.breite-1)
        y = np.clip(y, 0, self.hoehe-1)
        
        return {
            'nahrung': self.nahrungskarte[x, y],
            'terrain': self.terrain[x, y],
            'jahreszeit': self.jahreszeit
        }


class Genom:
    """Repräsentiert das Genom eines Organismus mit Mutationsfähigkeit."""
    
    def __init__(self, gen_groesse=10, mutationsrate=0.01, von_eltern=None):
        self.gen_groesse = gen_groesse
        self.mutationsrate = mutationsrate
        
        if von_eltern is None:
            # Zufälliges Genom erstellen
            self.gene = np.random.uniform(-1, 1, gen_groesse)
        else:
            # Genom von Eltern erben mit Mutation
            vater_genom, mutter_genom = von_eltern
            # Crossing-over
            crossover_punkt = np.random.randint(0, gen_groesse)
            self.gene = np.zeros(gen_groesse)
            self.gene[:crossover_punkt] = vater_genom.genom.gene[:crossover_punkt]
            self.gene[crossover_punkt:] = mutter_genom.genom.gene[crossover_punkt:]
            
            # Mutation
            for i in range(gen_groesse):
                if np.random.random() < self.mutationsrate:
                    self.gene[i] += np.random.normal(0, 0.2)  # Kleine Änderung
                    self.gene[i] = np.clip(self.gene[i], -1, 1)
    
    def get_eigenschaft(self, index):
        """Gibt die genetische Eigenschaft am gegebenen Index zurück."""
        return self.gene[index % self.gen_groesse]


class Gehirn:
    """Neuronales Netzwerk als 'Gehirn' des Organismus für Entscheidungsfindung."""
    
    def __init__(self, eingabe_groesse, ausgabe_groesse, hidden_schichten=(16, 8), genom=None):
        self.eingabe_groesse = eingabe_groesse
        self.ausgabe_groesse = ausgabe_groesse
        self.schichten_groesse = [eingabe_groesse] + list(hidden_schichten) + [ausgabe_groesse]
        
        # Initialisiere Gewichte und Bias
        self.gewichte = []
        self.bias = []
        
        # Wenn genom vorhanden, nutze es für die Initialisierung
        if genom is not None:
            gen_index = 0
            for i in range(len(self.schichten_groesse) - 1):
                n_input = self.schichten_groesse[i]
                n_output = self.schichten_groesse[i + 1]
                
                # Gewichte aus Genom extrahieren
                w_flach = []
                for j in range(n_input * n_output):
                    w_flach.append(genom.get_eigenschaft(gen_index))
                    gen_index += 1
                
                w = np.array(w_flach).reshape(n_input, n_output)
                self.gewichte.append(w)
                
                # Bias aus Genom extrahieren
                b = []
                for j in range(n_output):
                    b.append(genom.get_eigenschaft(gen_index))
                    gen_index += 1
                
                self.bias.append(np.array(b))
        else:
            # Zufällige Initialisierung falls kein Genom vorhanden
            for i in range(len(self.schichten_groesse) - 1):
                self.gewichte.append(
                    np.random.uniform(-0.5, 0.5, (self.schichten_groesse[i], self.schichten_groesse[i + 1]))
                )
                self.bias.append(
                    np.random.uniform(-0.5, 0.5, self.schichten_groesse[i + 1])
                )
    
    def aktivierungsfunktion(self, x):
        """ReLU-Aktivierungsfunktion."""
        return np.maximum(0, x)
    
    def ausgabe_aktivierung(self, x):
        """Sigmoid-Aktivierung für die Ausgabeschicht."""
        return 1.0 / (1.0 + np.exp(-x))
    
    def vorwaertsdurchlauf(self, eingabe):
        """Berechnet die Gehirnausgabe für die gegebene Eingabe."""
        aktivierung = np.array(eingabe)
        
        # Hidden Layers mit ReLU
        for i in range(len(self.gewichte) - 1):
            z = np.dot(aktivierung, self.gewichte[i]) + self.bias[i]
            aktivierung = self.aktivierungsfunktion(z)
        
        # Ausgabeschicht mit Sigmoid für normalisierte Ausgaben
        z_ausgabe = np.dot(aktivierung, self.gewichte[-1]) + self.bias[-1]
        ausgabe = self.ausgabe_aktivierung(z_ausgabe)
        
        return ausgabe


class Organismus:
    """Ein virtueller Organismus mit Genom, Phänotyp und Verhalten."""
    
    def __init__(self, umwelt, position=None, energie=100, genom=None, eltern=None, spezies_id=None):
        self.umwelt = umwelt
        self.alter = 0
        self.lebend = True
        
        # Position zufällig initialisieren falls nicht angegeben
        if position is None:
            self.position = np.array([
                np.random.uniform(0, umwelt.breite),
                np.random.uniform(0, umwelt.hoehe)
            ])
        else:
            self.position = np.array(position)
        
        self.energie = energie
        self.max_energie = 200
        self.alter_max = 500 + np.random.randint(-50, 50)  # Natürliche Varianz
        
        # Genom erstellen oder erben
        if genom is None and eltern is None:
            self.genom = Genom()
        elif eltern is not None:
            self.genom = Genom(von_eltern=eltern)
        else:
            self.genom = genom
        
        # Phänotyp aus Genom ableiten
        self._phenotyp_entwickeln()
        
        # Gehirn initialisieren
        eingabe_groesse = 8  # Position, Nahrung, Terrain, Energie, Jahreszeit, 3 Sensoren
        ausgabe_groesse = 4  # Bewegung (x,y), Fressen, Reproduzieren
        self.gehirn = Gehirn(eingabe_groesse, ausgabe_groesse, genom=self.genom)
        
        # Spezies zuordnen
        if spezies_id is None and eltern is not None:
            # Erbe Spezies von Eltern, wenn ähnlich genug
            self.spezies_id = eltern[0].spezies_id
            # Mögliche Speziation basierend auf genetischer Distanz
            if np.sum(np.abs(self.genom.gene - eltern[0].genom.gene)) > 3.0:
                self.spezies_id = np.random.randint(1000000)
        elif spezies_id is None:
            self.spezies_id = np.random.randint(1000000)
        else:
            self.spezies_id = spezies_id
        
        # Speichere Statistiken für Reinforcement Learning
        self.belohnungen = []
        self.aktionen = []
        self.zustande = []
    
    def _phenotyp_entwickeln(self):
        """Leitet den Phänotyp vom Genom ab."""
        # Basis-Eigenschaften
        self.groesse = 0.5 + 0.5 * (self.genom.get_eigenschaft(0) + 1) / 2  # 0.5 - 1.0
        self.geschwindigkeit = 1.0 + 2.0 * (self.genom.get_eigenschaft(1) + 1) / 2  # 1.0 - 3.0
        self.sichtweite = 10.0 + 20.0 * (self.genom.get_eigenschaft(2) + 1) / 2  # 10 - 30
        self.metabolismus = 0.5 + 1.0 * (self.genom.get_eigenschaft(3) + 1) / 2  # 0.5 - 1.5
        
        # Energiekosten skalieren mit Eigenschaften
        self.energie_pro_schritt = 0.1 + 0.1 * self.geschwindigkeit + 0.2 * self.groesse
        self.energie_reproduktion = 50.0 * self.groesse
    
    def wahrnehmen(self):
        """Sammelt Umweltinformationen für die Entscheidungsfindung."""
        # Eigene Position und Status
        normalisierte_position = self.position / np.array([self.umwelt.breite, self.umwelt.hoehe])
        
        # Umweltdaten an aktueller Position
        umwelt_daten = self.umwelt.get_umwelt_zustand(self.position)
        
        # Sensoren für Nahrung in der Umgebung (vorne, links, rechts)
        sensoren = []
        winkel = [0, np.pi/2, -np.pi/2]  # Vorne, links, rechts
        for w in winkel:
            # Berechne Sensorposition
            sensor_richtung = np.array([np.cos(w), np.sin(w)])
            sensor_pos = self.position + sensor_richtung * self.sichtweite
            sensor_pos = np.clip(sensor_pos, [0, 0], 
                                 [self.umwelt.breite-1, self.umwelt.hoehe-1])
            
            # Nahrungswert am Sensor
            sensor_umwelt = self.umwelt.get_umwelt_zustand(sensor_pos)
            sensoren.append(sensor_umwelt['nahrung'])
        
        # Gesamte Wahrnehmung zusammenstellen
        wahrnehmung = [
            normalisierte_position[0],  # x-Position
            normalisierte_position[1],  # y-Position
            umwelt_daten['nahrung'],    # Nahrung an Position
            umwelt_daten['terrain'],    # Terrain
            self.energie / self.max_energie,  # Energie
            umwelt_daten['jahreszeit'] / 3.0,  # Jahreszeit
            sensoren[0],  # Nahrung vorne
            sensoren[1],  # Nahrung links
            sensoren[2]   # Nahrung rechts
        ]
        
        return np.array(wahrnehmung[:self.gehirn.eingabe_groesse])  # Auf Eingabegröße anpassen
    
    def entscheiden(self, wahrnehmung):
        """Trifft Entscheidungen basierend auf der Wahrnehmung."""
        return self.gehirn.vorwaertsdurchlauf(wahrnehmung)
    
    def handeln(self, entscheidung):
        """Führt Aktionen basierend auf der Entscheidung aus."""
        # Entscheidung zerlegen
        bewegung_x = (entscheidung[0] - 0.5) * 2  # -1 bis 1
        bewegung_y = (entscheidung[1] - 0.5) * 2  # -1 bis 1
        fressen = entscheidung[2] > 0.5
        reproduzieren = entscheidung[3] > 0.5
        
        belohnung = 0
        
        # Bewegung ausführen
        bewegung = np.array([bewegung_x, bewegung_y])
        norm = np.linalg.norm(bewegung)
        if norm > 0:
            bewegung = bewegung / norm  # Normalisieren
        
        # Geschwindigkeit anwenden und Energiekosten berechnen
        self.position += bewegung * self.geschwindigkeit
        self.position = np.clip(self.position, [0, 0], 
                               [self.umwelt.breite-1, self.umwelt.hoehe-1])
        
        # Energiekosten für Bewegung
        self.energie -= self.energie_pro_schritt
        
        # Fressen, wenn möglich
        if fressen:
            umwelt_daten = self.umwelt.get_umwelt_zustand(self.position)
            nahrungsmenge = umwelt_daten['nahrung']
            
            if nahrungsmenge > 0.1:  # Mindestmenge an Nahrung
                # Nahrung konsumieren (max 0.5 pro Zeitschritt)
                verbrauchte_nahrung = min(0.5, nahrungsmenge)
                self.umwelt.nahrungskarte[int(self.position[0]), int(self.position[1])] -= verbrauchte_nahrung
                
                # Energie gewinnen
                energie_gewinn = verbrauchte_nahrung * 50.0 * self.groesse
                self.energie += energie_gewinn
                self.energie = min(self.energie, self.max_energie)
                
                # Belohnung für erfolgreiche Nahrungsaufnahme
                belohnung += energie_gewinn / 10.0
        
        # Reproduzieren, wenn genug Energie
        nachkommen = []
        if reproduzieren and self.energie > self.energie_reproduktion:
            # Energiekosten
            self.energie -= self.energie_reproduktion
            
            # Position für Nachkommen
            nachkommen_pos = self.position + np.random.uniform(-3, 3, 2)
            nachkommen_pos = np.clip(nachkommen_pos, [0, 0], 
                                    [self.umwelt.breite-1, self.umwelt.hoehe-1])
            
            # Nachkommen erzeugen (benötigt einen Partner für sexuelle Reproduktion)
            # Hier vereinfacht als Selbstreproduktion
            nachkommen.append(
                Organismus(
                    self.umwelt,
                    position=nachkommen_pos,
                    energie=self.energie_reproduktion * 0.5,
                    eltern=(self, self)
                )
            )
            
            # Belohnung für Reproduktion
            belohnung += 5.0
        
        # Altern
        self.alter += 1
        if self.alter >= self.alter_max or self.energie <= 0:
            self.lebend = False
            belohnung -= 10.0  # Negative Belohnung für Sterben
        
        return belohnung, nachkommen
    
    def update(self):
        """Führt einen kompletten Aktualisierungszyklus aus."""
        if not self.lebend:
            return []
        
        # 1. Wahrnehmen
        wahrnehmung = self.wahrnehmen()
        self.zustande.append(wahrnehmung)
        
        # 2. Entscheiden
        entscheidung = self.entscheiden(wahrnehmung)
        self.aktionen.append(entscheidung)
        
        # 3. Handeln
        belohnung, nachkommen = self.handeln(entscheidung)
        self.belohnungen.append(belohnung)
        
        return nachkommen


class EvolutionsSimulation:
    """Hauptklasse für die Evolutionssimulation."""
    
    def __init__(self, breite=100, hoehe=100, start_population=50):
        self.umwelt = Umwelt(breite, hoehe)
        self.organismen = []
        self.generationen = 0
        self.zeit = 0
        
        # Anfangspopulation erzeugen
        for _ in range(start_population):
            self.organismen.append(Organismus(self.umwelt))
        
        # Statistikdaten
        self.statistik = {
            'population': [start_population],
            'arten': [len(self._get_arten())],
            'durchschn_energie': [np.mean([o.energie for o in self.organismen])],
            'durchschn_groesse': [np.mean([o.groesse for o in self.organismen])],
            'durchschn_geschwindigkeit': [np.mean([o.geschwindigkeit for o in self.organismen])]
        }
    
    def _get_arten(self):
        """Ermittelt die aktuellen Arten in der Simulation."""
        arten = defaultdict(int)
        for org in self.organismen:
            arten[org.spezies_id] += 1
        return arten
    
    def update(self):
        """Führt einen Simulationsschritt aus."""
        self.zeit += 1
        
        # Umwelt aktualisieren
        self.umwelt.aktualisieren()
        
        # Organismen aktualisieren und neue Nachkommen sammeln
        neue_organismen = []
        for org in self.organismen:
            nachkommen = org.update()
            neue_organismen.extend(nachkommen)
        
        # Tote Organismen entfernen
        self.organismen = [org for org in self.organismen if org.lebend]
        
        # Neue Nachkommen hinzufügen
        self.organismen.extend(neue_organismen)
        
        # Statistiken aktualisieren
        if len(self.organismen) > 0:
            self.statistik['population'].append(len(self.organismen))
            self.statistik['arten'].append(len(self._get_arten()))
            self.statistik['durchschn_energie'].append(np.mean([o.energie for o in self.organismen]))
            self.statistik['durchschn_groesse'].append(np.mean([o.groesse for o in self.organismen]))
            self.statistik['durchschn_geschwindigkeit'].append(np.mean([o.geschwindigkeit for o in self.organismen]))
        else:
            # Bei Aussterben Nullen eintragen
            for key in self.statistik:
                self.statistik[key].append(0)
        
        # Zufällige neue Organismen einfügen, falls Population kritisch sinkt
        if len(self.organismen) < 10:
            for _ in range(5):
                self.organismen.append(Organismus(self.umwelt))
    
    def run(self, schritte=1000, visualisieren=True):
        """Führt die Simulation für eine bestimmte Anzahl von Schritten aus."""
        if visualisieren:
            return self._run_mit_visualisierung(schritte)
        else:
            return self._run_ohne_visualisierung(schritte)
    
    def _run_ohne_visualisierung(self, schritte):
        """Führt die Simulation ohne visuelle Ausgabe aus."""
        start_zeit = time.time()
        for _ in range(schritte):
            self.update()
        
        # Endstatistiken ausgeben
        ende_zeit = time.time()
        durchlaufzeit = ende_zeit - start_zeit
        print(f"Simulation abgeschlossen: {schritte} Schritte in {durchlaufzeit:.2f}s")
        print(f"Finale Population: {len(self.organismen)}")
        print(f"Anzahl Arten: {len(self._get_arten())}")
        
        return self.statistik
    
    def _run_mit_visualisierung(self, schritte):
        """Führt die Simulation mit Visualisierung aus."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Farbkarte für die Nahrung
        nahrung_img = ax1.imshow(self.umwelt.nahrungskarte.T, 
                                 cmap='YlGn', origin='lower', 
                                 extent=[0, self.umwelt.breite, 0, self.umwelt.hoehe])
        
        # Organismen als Scatter-Plot
        organismen_scatter = ax1.scatter([], [], c=[], s=[], alpha=0.8)
        
        # Statistische Grafiken
        stat_linien = {}
        for key in ['population', 'arten', 'durchschn_energie', 'durchschn_groesse', 'durchschn_geschwindigkeit']:
            stat_linien[key], = ax2.plot([], [], label=key)
        
        ax2.legend()
        ax2.set_xlabel('Zeit')
        ax2.set_title('Evolutionsstatistiken')
        
        # Aktualisierungsfunktion für die Animation
        def update_frame(frame):
            # Simulationsschritt ausführen
            self.update()
            
            # Nahrungskarte aktualisieren
            nahrung_img.set_array(self.umwelt.nahrungskarte.T)
            
            # Organismen aktualisieren
            if self.organismen:
                pos = np.array([org.position for org in self.organismen])
                farben = [f"C{org.spezies_id % 10}" for org in self.organismen]
                groessen = [30 * org.groesse for org in self.organismen]
                
                organismen_scatter.set_offsets(pos)
                organismen_scatter.set_color(farben)
                organismen_scatter.set_sizes(groessen)
            else:
                organismen_scatter.set_offsets(np.empty((0, 2)))
            
            # Statistiken aktualisieren
            for key, line in stat_linien.items():
                line.set_data(range(len(self.statistik[key])), self.statistik[key])
                ax2.relim()
                ax2.autoscale_view()
            
            titel = f"Evolutionssimulation - Zeit: {self.zeit}, Pop: {len(self.organismen)}, Arten: {len(self._get_arten())}"
            ax1.set_title(titel)
            
            return nahrung_img, organismen_scatter, *stat_linien.values()
        
        # Animation erstellen und starten
        ani = FuncAnimation(fig, update_frame, frames=schritte, blit=True, interval=50)
        plt.tight_layout()
        plt.show()
        
        return self.statistik


def run_parallel_simulation(param_sets, schritte=1000):
    """Führt mehrere Simulationen parallel aus mit verschiedenen Parametern."""
    def worker(params):
        breite = params.get('breite', 100)
        hoehe = params.get('hoehe', 100)
        pop = params.get('population', 50)
        
        sim = EvolutionsSimulation(breite, hoehe, pop)
        stats = sim._run_ohne_visualisierung(schritte)
        return stats, params
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        ergebnisse = pool.map(worker, param_sets)
    
    return ergebnisse


if __name__ == "__main__":
    # Beispiel: Einzelne Simulation starten
    sim = EvolutionsSimulation(breite=100, hoehe=100, start_population=50)
    sim.run(schritte=500, visualisieren=True)
    
    # Beispiel: Parallele Simulationen mit verschiedenen Parametern
    """
    param_sets = [
        {'breite': 100, 'hoehe': 100, 'population': 30},
        {'breite': 100, 'hoehe': 100, 'population': 60},
        {'breite': 150, 'hoehe': 150, 'population': 50}
    ]
    ergebnisse = run_parallel_simulation(param_sets, 1000)
    
    # Ergebnisse visualisieren
    plt.figure(figsize=(10, 6))
    for stats, params in ergebnisse:
        label = f"Pop: {params['population']}, Größe: {params['breite']}x{params['hoehe']}"
        plt.plot(stats['population'], label=label)
    
    plt.legend()
    plt.xlabel('Zeit')
    plt.ylabel('Population')
    plt.title('Populationsentwicklung über verschiedene Parameter')
    plt.show()
    """
