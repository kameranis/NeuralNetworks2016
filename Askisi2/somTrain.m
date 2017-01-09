function somTrain(patterns)
% συνάρτηση που εκπαιδεύει ένα SOM
%
% INPUT:
% patterns πίνακας DxP που περιέχει το σύνολο των προτύπων εκπαίδευσης για
% το SOM (training set).
%
% Η εν λόγω συνάρτηση εκπαιδεύει ένα SOM παρουσιάζοντας κάθε πρότυπο,
% προσδιορίζοντας τον νευρώνα νικητή (ανταγωνισμός), υπολογίζοντας τις
% ενεργοποιήσεις (συνεργασία) και τροποποιώντας/ανανεώνοντας τις παραμέτρους
% του SOM βάσει του αλγορίθμου μάθησης του Kohonen (ανταμοιβή). Αυτή η
% διαδικασία γίνεται ακολουθιακά και ξεχωριστά για τα πρότυπα κάθε εποχής
% εκπαίδευσης. Γενικότερα, η εκπαίδευση περιλαμβάνει αμφότερα τα στάδια ordering
% και tuning, καθένα εκ των οποίων διαρκεί έναν ορισμένο αριθμό εποχών.
% Το στάδιο ordering διαρκεί orderSteps εποχές. Η απόσταση εντός της οποίας
% ο νευρώνας νικητής και κάθε άλλος νευρώνας θεωρούνται γειτονικοί, ξεκινάει ως η
% μέγιστη απόσταση μεταξύ δυο οποιωνδήποτε νευρώνων (μεταβλητή
% maxNeighborDist) και μειώνεται (εκθετικά) μέχρι την τιμή tuneND. Αντίστοιχα, η
% αρχική τιμή του ρυθμού μάθησης είναι orderLR και μειώνεται ομοίως (εκθετικά) μέχρι
% την τιμή tuneLR.
% Το στάδιο tuning διαρκεί σαφώς περισσότερες εποχές από ό,τι το στάδιο
% ordering (συνήθως κατά ένα συντελεστή από 2 έως 5). Η απόσταση εντός της
% οποίας ο νευρώνας νικητής και κάθε άλλος νευρώνας θεωρούνται γειτονικοί είναι
% σταθερή και ίση με tuneND. Ο ρυθμός μάθησης είτε διατηρείται αμετάβλητος και
% ίσος με tuneLR, είτε με αρχική τιμή την tuneLR μειώνεται με ιδιαίτερα αργό τρόπο.

global maxNeighborDist orderSteps orderLR tuneND tuneLR;

for i=1:orderSteps
    % exponential decay for neighbor distance and learning rate
    neighborDist = maxNeighborDist * exp((i-1)/(orderSteps-1) * log(tuneND/maxNeighborDist));
    learningRate = orderLR * exp((i-1)/(orderSteps-1) * log(tuneLR/orderLR));
    for k=1:size(patterns, 2)
        somUpdate(patterns(:,k), learningRate, neighborDist)
    end
end

for i=1:4*orderSteps
    for k=1:length(patterns)
        somUpdate(patterns(:,k), learningRate, neighborDist)
    end
end

end

