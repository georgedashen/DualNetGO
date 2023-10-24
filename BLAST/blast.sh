makeblastdb -in human_P_train.fa -dbtype prot -out human_P_train_prot_set && \ 
        blastp -db human_P_train_prot_set -query human_P_test.fa -out blast_result_human_P.txt -outfmt 6 & #tabular
makeblastdb -in human_F_train.fa -dbtype prot -out human_F_train_prot_set && \
        blastp -db human_F_train_prot_set -query human_F_test.fa -out blast_result_human_F.txt -outfmt 6 & #tabular
makeblastdb -in human_C_train.fa -dbtype prot -out human_C_train_prot_set && \
        blastp -db human_C_train_prot_set -query human_C_test.fa -out blast_result_human_C.txt -outfmt 6 & #tabular
makeblastdb -in mouse_P_train.fa -dbtype prot -out mouse_P_train_prot_set && \
        blastp -db mouse_P_train_prot_set -query mouse_P_test.fa -out blast_result_mouse_P.txt -outfmt 6 & #tabular
makeblastdb -in mouse_F_train.fa -dbtype prot -out mouse_F_train_prot_set && \
        blastp -db mouse_F_train_prot_set -query mouse_F_test.fa -out blast_result_mouse_F.txt -outfmt 6 & #tabular
makeblastdb -in mouse_C_train.fa -dbtype prot -out mouse_C_train_prot_set && \
        blastp -db mouse_C_train_prot_set -query mouse_C_test.fa -out blast_result_mouse_C.txt -outfmt 6 & #tabular

wait
