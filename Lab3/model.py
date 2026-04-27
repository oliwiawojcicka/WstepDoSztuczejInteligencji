import math


class DecisionTreeClassifier():

    def __init__(self, max_depth: int):
        """
        max_depth — maksymalna dopuszczalna głębokość drzewa.
        Jeśli depth == max_depth → tworzony jest liść, dalszy podział nie jest wykonywany.
        """
        self.max_depth = max_depth
        self.tree = None

    @staticmethod
    def entropy(labels):
        """
        Przyjmuje: labels — lista klas [0, 1, 1, 0, 1, ...]
        Entropia zbioru U:
            I(U) = - Σ f_i * log(f_i)
        gdzie f_i to częstość klasy i w danym zbiorze
        """
        n = len(labels)
        if n == 0:
            return 0.0
        entropy_value = 0.0
        unique_labels = set(labels)
        for cls in unique_labels:
            count = labels.count(cls)
            f = count / n
            entropy_value -= f * math.log(f)
        return entropy_value

    def inf_gain(self, X, y, attribute_index):
        """
        Przyjmuje: X — lista przykładów, y — lista etykiet klas, attribute_index — numer kolumny (cechy), po której robimy podział
        Entropia po podziale wg atrybutu d:
            Inf(d, U) = Σ_j  ( |U_j| / |U| ) * I(U_j)
        Zdobycz informacyjna:
            InfGain(d, U) = I(U) - Inf(d, U)
        """
        n = len(X)
        if n == 0:
            return 0.0
        base_entropy = self.entropy(y) #I(U)
        attr_values = set(row[attribute_index] for row in X) #unikalne wartości badanego atrybutu
        weighted_entropy_after_split = 0.0 #To będzie: Σ_j ( |U_j| / |U| ) * I(U_j )
        #Obliczamy Inf(d,U)
        for val in attr_values:
            #Podzbiór U_j
            X_sub = []
            y_sub = []
            for row, label in zip(X, y):
                if row[attribute_index] == val:
                    X_sub.append(row)
                    y_sub.append(label)
            if len(X_sub) == 0:
                continue
            weight = len(X_sub) / n #|U_j| / |U|
            weighted_entropy_after_split += weight * self.entropy(y_sub)
        return base_entropy - weighted_entropy_after_split #InfGain(d,U) = I(U) - Inf(d,U)

    @staticmethod
    def majority_class(labels):
        """
        Zwraca najczęściej występującą etykietę (0 lub 1).
        Jeśli y = [1,1,0,1,0,1] → zwróci 1.
        """
        if not labels:
            return None
        unique = set(labels)
        best_label = None
        best_count = -1
        for cls in unique:
            cnt = labels.count(cls)
            if cnt > best_count:
                best_count = cnt
                best_label = cls
        return best_label

    def ID3(self, X, y, remaining_attributes, depth):
        """
        Przyjmuje: X — lista przykładów, y — lista ich etykiet
        remaining_attributes — lista indeksów cech, których wolno użyć, depth — bieżąca głębokość
        REKURENCYJNIE buduje drzewo:
          - sprawdza warunki stopu
          - wybiera najlepszy atrybut
          - dzieli dane
          - wywołuje ID3 osobno dla każdej wartości atrybutu
        """
        #Warunek1: puste dane (U == ∅) -> lisc
        if not X or not y:
            return {"leaf": True, "label": None}

        #Warunek2: wszystkie etykiety takie same (∀(x_i, y_i) ∈ U: y_i == y) -> lisc z tą klasą
        if len(set(y)) == 1:
            return {"leaf": True, "label": y[0]}

        #Warunek3: osiągnięto max_depth -> lisc z najczestsza klasą
        if depth >= self.max_depth:
            return {"leaf": True, "label": self.majority_class(y)}

        #Warunek4: brak cech do podziału (|D| == 0) -> lisc z najczestszą klasą
        if not remaining_attributes:
            return {"leaf": True, "label": self.majority_class(y)}

        #Wybór najlepszego atrybutu d = argmax_d InfGain(d, U)
        best_attr = None
        best_gain = -1.0
        for attr in remaining_attributes:
            gain = self.inf_gain(X, y, attr)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr

        #Jeśli brak poprawy — liść
        if best_gain <= 0:
            return {"leaf": True, "label": self.majority_class(y)}

        #Tworzymy węzeł drzewa
        node = {
            "leaf": False,
            "attribute": best_attr,
            "children": {},
            "label": self.majority_class(y)
        }

        #Podział danych na grupy (U_j = { (x_i, y_i) ∈ U : x_i[d] = d_j })
        attr_values = set(row[best_attr] for row in X)
        next_attributes = [a for a in remaining_attributes if a != best_attr]
        for val in attr_values:
            X_sub = []
            y_sub = []
            for row, label in zip(X, y):
                if row[best_attr] == val:
                    X_sub.append(row)
                    y_sub.append(label)
            #Rekurencyjna budowa poddrzewa (drzewo(d, ID3(U_1), ID3(U_2), ...))
            child_node = self.ID3(X_sub, y_sub, next_attributes, depth + 1)
            node["children"][val] = child_node
        return node

    def fit(self, X, y):
        """
        Buduje drzewo decyzyjne z pełnego zbioru treningowego
        """
        num_features = len(X[0])  # liczba cech
        attributes = list(range(num_features))  # indeksy: [0,1,...]
        self.tree = self.ID3(X, y, attributes, depth=0)
        return self

    def predict(self, X):
        """
        Zwraca listę przewidywanych klas
        """
        if self.tree is None:
            raise ValueError("Model nie został wytrenowany (tree == None).")
        predictions = []
        for row in X:
            node = self.tree
            # schodzimy w dół dopóki nie trafimy na liść
            while not node["leaf"]:
                attr = node["attribute"]
                value = row[attr]
                if value in node["children"]:
                    node = node["children"][value]
                else:
                    # brak odpowiedniej gałęzi
                    break
            predictions.append(node["label"])
        return predictions
