
-- NON SPATIAL QUERY MX1
-- MX1. Χρήση του τελεστή συνάθροισης SUM() σε στοιχεία που έχουν ομαδοποιηθεί με τον τελεστή ομαδοποίησης GROUP_BY()
-- Ερώτημα: Να υπολογιστεί η συνολική απόδοση ενός συγκεκριμένου αγροκτήματος σε όλα τα είδη καλλιέργειας. Τα αποτελέσματα να ταξινομηθούν σε φθίνουσα σειρά με βάση τη συνολική απόδοση.
SELECT parcel_id, SUM(performance) AS total_performance
FROM annual_performances
GROUP BY parcel_id
ORDER BY SUM(performance) DESC


-- NON SPATIAL QUERY MX2
-- MX2. Χρήση του τελεστή συνάθροισης COUNT() σε στοιχεία που έχουν ομαδοποιηθεί με τον τελεστή ομαδοποίησης GROUP_BY(). Χρήση του τελεστή συνάθροισης MAX(). Χρήση εμφωλευμένου ερωτήματος.
-- Ερώτημα: Πόσοι είναι οι περισσότεροι άνθρωποι που απασχολούνται σε κάποιο αγρόκτημα.
SELECT MAX(employees)
FROM(
	SELECT parcel_id, COUNT(*) AS employees
	FROM people
	GROUP BY parcel_id
	ORDER BY COUNT(*) DESC
	) AS T


-- NON SPATIAL QUERY MX3
-- MX3. Χρήση του τελεστή συνάθροισης MIN(). Χρήση εμφωλευμένου ερωτήματος.
-- Ερώτημα: Ποιο είναι το αγρόκτημα με τη μικρότερη (μη μηδενική) αξία ανά στρέμμα.
SELECT parcel_id, asmt/acres AS dols_per_acre
FROM parcels
WHERE asmt/acres = (SELECT MIN(T.dols_per_acre)
					FROM(SELECT asmt/acres AS dols_per_acre
						 FROM parcels
						 WHERE asmt/acres <> 0) AS T)


-- NON SPATIAL QUERY MX4
-- MX4. Χρήση των τελεστών συνάθροισης AVG() και COUNT(). Χρήση εμφωλευμένου ερωτήματος.
-- Ερώτημα: Πόσα είναι τα αγρόκτηματα αξία ανά στρέμμα (μη μηδενική) μεγαλύτερη από το μέσο όρο.
SELECT COUNT(*)
FROM parcels
WHERE asmt/acres > (SELECT AVG(T.dols_per_acre)
					FROM(SELECT asmt/acres AS dols_per_acre
						 FROM parcels
						 WHERE asmt/acres <> 0) AS T)


-- NON SPATIAL QUERY MX5
-- MX5. Χρήση ερωτήματος σύνδεσης JOIN. Χρήση του τελεστή συνάθροισης COUNT() σε στοιχεία που έχουν ομαδοποιηθεί με τον τελεστή ομαδοποίησης GROUP_BY(). Χρήση εμφωλευμένου ερωτήματος.
-- Ερώτημα: Πόσα είναι τα αγρόκτηματα που ασχολούνται με την καλλιέργεια του κάθε φυτού κάθε εποχή του χρόνου.
SELECT p.season, pcp.plant_id, COUNT(*) AS num_of_parcels
FROM parcels pc JOIN parc_cult_plants pcp ON pc.parcel_id = pcp.parcel_id JOIN plants p ON pcp.plant_id = p.name
GROUP BY pcp.plant_id, p.season
ORDER BY p.season, pcp.plant_id ASC


-- NON SPATIAL QUERY MX6
-- MX6. Χρήση ερωτήματος σύνδεσης JOIN. Χρήση του τελεστή συνάθροισης AVG() σε στοιχεία που έχουν ομαδοποιηθεί με τον τελεστή ομαδοποίησης GROUP_BY(). Χρήση εμφωλευμένου ερωτήματος.
-- Ερώτημα: Ποια εποχή ευδοκιμεί το φυτό με την καλύτερη απόδοση καλλιέργειας.
SELECT season
FROM (
	SELECT ap.plant_id, p.season, AVG(performance) AS avg_perf
	FROM plants p JOIN annual_performances ap ON p.name = ap.plant_id
	GROUP BY ap.plant_id, p.season
	ORDER BY AVG(performance) DESC
	) AS T
LIMIT 1


-- NON SPATIAL QUERY MX7
-- MX7. Χρήση εμφωλευμένου ερωτήματος πολλαπλών επιπέδων. Χρήση του τελεστή συνάθροισης AVG() και του τελεστή συνάθροισης SUM() σε στοιχεία που έχουν ομαδοποιηθεί με τον τελεστή ομαδοποίησης GROUP_BY(). Χρήση του τελεστή IN() για να αποδοθούν πολλαπλές συγκεκριμένες τιμές στο πεδίο WHERE. Χρήση του τελεστή :: για αλλαγή του τύπου δεδομένων. Χρήση του τελεστή χαρακτήρα Left() για να απομονωθεί ένα συγκεκριμένο αριστερό τμήμα ενός string.
-- Ερώτημα: Να υπολογιστεί η μέση απόδοση καλλιέργειας του φυτού plant_15 στα αγροκτήματα που το id τους ξεκινά με '100'.
SELECT AVG(sum_perf) AS average_performance
FROM (SELECT parcel_id, SUM(performance) AS sum_perf
		FROM annual_performances
		WHERE parcel_id IN (SELECT parcel_id
							 FROM parcels
							 WHERE Left(parcel_id::text,3) = '100'
							 )
	  			AND plant_id = 'plant_15'
		GROUP BY parcel_id) AS T


-- NON SPATIAL QUERY MX8
-- MX8. Δημιουργία και χρήση View για την προσωρινή αποθήκευση χρήσιμων στοιχείων. Χρήση του τελεστή συνάθροισης COUNT() σε στοιχεία που έχουν ομαδοποιηθεί με τον τελεστή ομαδοποίησης GROUP_BY(). Χρήση του τελεστή IN() για να αποδοθούν πολλαπλές συγκεκριμένες τιμές στο πεδίο WHERE. Χρήση εμφωλευμένου ερωτήματος.
-- Ερώτημα: Να υπολογιστεί το πλήθος των ατόμων που δουλεύουν σε κάθε ένα από τα 5 αγροκτήματα, που παρουσιάζουν τις μεγαλύτερες ετήσιες αποδόσεις.

-- ΜΧ8.1 Δημιουργία όψης όπου αποθηκεύονται τα στοιχεία (id και απόδοση) των 5 πιο αποδοτικών αγροκτημάτων
-- DROP VIEW max_perf CASCADE;
CREATE VIEW max_perf AS
	SELECT parcel_id, SUM(performance) AS total_performance
	FROM annual_performances
	GROUP BY parcel_id
	ORDER BY SUM(performance) DESC
	LIMIT 5;
-- SELECT * FROM max_perf

-- ΜΧ8.2 Ανάκτηση των τελικών στοιχείων του ερωτήματος με χρήση των στοιχείων που είναι αποθηκευμένα στην παραπάνω όψη.
SELECT parcel_id, COUNT(*)
FROM people
WHERE parcel_id IN (SELECT parcel_id
				   	 FROM max_perf)
GROUP BY parcel_id




