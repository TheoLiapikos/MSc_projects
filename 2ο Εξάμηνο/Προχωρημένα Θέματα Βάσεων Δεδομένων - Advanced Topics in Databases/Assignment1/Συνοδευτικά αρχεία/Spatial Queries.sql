

-- SPATIAL QUERY X1
-- X1.1 Χρήση του ST_Area και ST_Union για να υπολογιστεί το συνολικό εμβαδόν των 5 πιο ακριβών parcels του χάρτη
-- Αρχικά πέρνω τις γεωμετρίες για τις εμφανίσω στο χάρτη
DROP TABLE query_layer;
SELECT parcels.geometry
INTO query_layer
FROM parcels
ORDER BY parcels.asmt DESC
LIMIT 5

-- X1.2 Τα 5 εμβαδά μεμονωμένα
SELECT ST_Area(parcels.geometry)
FROM parcels
ORDER BY parcels.asmt DESC
LIMIT 5

-- X1.3 Δεν θέλω απλά να αθροίσω τα 5 εμβαδά. Εκμεταλλευόμενος τη συνάρτηση ST_Union, εννοποιώ τις 5 γεωμετρίες
-- σε μία ενιαία και τέλος υπολογίζω το συνολικό εμβαδόν της
SELECT ST_Area(
				(SELECT ST_UNION(
								ARRAY(
									SELECT parcels.geometry
									FROM parcels
									ORDER BY parcels.asmt DESC
									LIMIT 5
									)
								)
				)
			)



-- SPATIAL QUERY X2
-- Χρήση του ST_Perimeter για να υπολογιστεί η περίμετρος του πιο μεγάλου πάρκου
-- X2.1 Αρχικά πέρνω όνομα και γεωμετρία για την εμφανίσω στο χάρτη
DROP TABLE query_layer;
SELECT name, parks.geometry
INTO query_layer
FROM parks
ORDER BY parks.size DESC
LIMIT 1

-- X2.2 Υπολογίζω την περίμετρο της επιστρεφόμενης γεωμετρίας
SELECT ST_Perimeter(T.geometry)
FROM(
    SELECT parks.geometry
    FROM parks
    ORDER BY parks.size DESC
    LIMIT 1) AS T



-- SPATIAL QUERY X3
-- ΕΜΠΕΡΙΕΧΕΤΑΙ (πλήρως) σε μια γεωμετρία
DROP TABLE query_layer;
SELECT parcels.*
INTO query_layer
FROM parcels, firm
WHERE ST_Contains(firm.geometry, parcels.geometry)
        AND firm.zone = 'X500'



-- SPATIAL QUERY X4
-- Σε αντιδιαστολή με απλά να το ΤΕΜΝΕΙ που επιστρέφει περισσότερα αποτελέσματα
DROP TABLE query_layer;
SELECT parcels.*
INTO query_layer
FROM parcels, firm
WHERE ST_Intersects(firm.geometry, parcels.geometry)
        AND firm.zone = 'X500'



-- SPATIAL QUERY X5
-- Χρήση OVERLAPS για να βρω ποια parcels ανήκουν μόνο εν μέρει στη ζώνη Χ500 του firm
DROP TABLE query_layer;
SELECT parcels.*
INTO query_layer
FROM parcels, firm
WHERE ST_Overlaps(firm.geometry, parcels.geometry)
        AND firm.zone = 'X500'



-- SPATIAL QUERY X6
-- Χρήση CROSSES για να βρω από ποια Πάρκα διέρχεται κάποιος δρόμος
DROP TABLE query_layer;
SELECT parks.geometry
INTO query_layer
FROM parks, roads
WHERE ST_Crosses(parks.geometry,roads.geometry)



-- SPATIAL QUERY X7
-- Χρήση ΕΜΦΩΛΕΥΜΕΝΟΥ ερωτήματος για τον υπολογισμό της μέγιστης απόστασης ανάμεσα στα Πάρκα
SELECT park1_name, max(parks_dist) AS max_parks_dist
FROM (
	SELECT p1.name AS park1_name, p2.name AS park2_name, ST_Distance(p1.geometry, p2.geometry) AS parks_dist
	FROM parks AS p1, parks AS p2
	WHERE p1.name <> p2.name
	ORDER BY p1.name, parks_dist ASC
	) AS PD
GROUP BY park1_name



-- SPATIAL QUERY X8
-- Χρήση VIEWS για τον υπολογισμό των ελάχιστων αποστάσεων ανάμεσα στα Πάρκα
-- X8.1 Δημιουργία View που περιέχει τις αποστάσεις του κάθε πάρκου από κάθε άλλο πάρκο. 
-- DROP VIEW Parks_Distances CASCADE;
CREATE VIEW Parks_Distances AS
    SELECT p1.name AS park1_name, p2.name AS park2_name, ST_Distance(p1.geometry, p2.geometry) AS parks_dist
    FROM parks AS p1, parks AS p2
    WHERE p1.name <> p2.name
    ORDER BY p1.name, parks_dist ASC;
-- SELECT * FROM Parks_Distances


-- X8.2 Δημιουργία View που περιέχει την ελάχιστη απόσταση του κάθε πάρκου από κάποιο άλλο πάρκο. 
-- DROP VIEW Min_Parks_Distance CASCADE;
CREATE VIEW Min_Parks_Distance AS
	SELECT park1_name, MIN(parks_dist) AS min_parks_dist
	FROM Parks_Distances
	GROUP BY park1_name;
-- SELECT * FROM Min_Parks_Distance


-- Χ8.3 Χρήση των παραπάνω Views για τον υπολογισμό της ελάχιστης απόστασης κάθε πάρκου από κάποιο άλλο πάρκο, μαζί με τα ονόματα αυτών.
SELECT minpd.park1_name, pd.park2_name, minpd.min_parks_dist
FROM Min_Parks_Distance AS minpd, Parks_Distances AS pd
WHERE minpd.min_parks_dist = pd.parks_dist AND minpd.park1_name <> pd.park2_name
ORDER BY minpd.min_parks_dist





