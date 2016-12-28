function somUpdate(pattern,learningRate,neighborDist)
% ενημέρωση/ανανέωση του συνόλου των παραμέτρων ενός SOM
%
% INPUT:
% pattern πίνακας Dx1 που περιέχει διατεταγμένα τα χαρακτηριστικά ενός
% προτύπου εισόδου, δηλαδή ουσιαστικά είναι ένα διάνυσμα στήλη διάστασης D.
%
% learningRate ο ρυθμός μάθησης κατά το παρόν βήμα της τροποποίησης
% των βαρών του SOM.
%
% neighborDist η απόσταση (στο πλέγμα του SOM) εντός της οποίας εάν
% βρίσκεται ένας νευρώνας θεωρείται γειτονικός του νευρώνα νικητή.
% Αυτή η συνάρτηση τροποποιεί/ενημερώνει τα βάρη (ή αλλιώς τις
% παραμέτρους) ενός SOM βάσει του αλγορίθμου μάθησης του Kohonen.
% Επιγραμματικά ο κανόνας μάθησης του Kohonen είναι ο εξής:
% Δwi = ηai(x-wi)
% όπου ο δείκτης i υποδεικνύει κάθε νευρώνα του SOM, wi το διάνυσμα παραμέτρων
% του νευρώνα i, x το πρότυπο εισόδου, ai η ενεργοποίηση του νευρώνα i και η ο
% ρυθμός μάθησης.

global IW N;

activatedNeurons = somActivation(pattern, neighborDist);
for i=1:N
    IW(i,:) = IW(i,:) + learningRate*activatedNeurons(i)*(pattern' - IW(i,:));
end

end
