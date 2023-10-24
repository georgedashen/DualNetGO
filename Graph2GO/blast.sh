makeblastdb -in human.fa -dbtype prot -out human_prot_set && \ 
	blastp -db human_prot_set -query human.fa -out sim_result_human.txt -outfmt 6 -max_target_seqs 30000 &
makeblastdb -in mouse.fa -dbtype prot -out mouse_prot_set && \
        blastp -db mouse_prot_set -query mouse.fa -out sim_result_mouse.txt -outfmt 6 -max_target_seqs 30000 &
