python network_preprocess.py --evidence neighborhood && python network_preprocess.py --evidence neighborhood --org mouse &
python network_preprocess.py --evidence cooccurence && python network_preprocess.py --evidence cooccurence --org mouse &
python network_preprocess.py --evidence coexpression && python network_preprocess.py --evidence coexpression --org mouse &
python network_preprocess.py --evidence experimental && python network_preprocess.py --evidence experimental --org mouse &
python network_preprocess.py --evidence database && python network_preprocess.py --evidence database --org mouse &
python network_preprocess.py --evidence textmining && python network_preprocess.py --evidence textmining --org mouse && python network_preprocess.py --evidence fusion --org mouse &

wait
