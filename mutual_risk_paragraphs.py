import os
import json
import pickle
import torch
import faiss
import numpy as np

def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

def search_index(index, query_embedding, k):
    distances, indices = index.search(query_embedding, k)
    return indices, distances

def main():
    id_to_paragraph = {}
    
    directory1 = "/home/ythsiao/fin.relation/output"
    firm_list = os.listdir(directory1)

    for firm in firm_list:
        directory2 = os.path.join(directory1, firm)
        directory3 = os.path.join(directory2, "10-K")
        tenK_list = os.listdir(directory3)
        tenK_list = sorted(tenK_list)

        for tenK in tenK_list:
            if (tenK[:4] == "2023") or (tenK[:4] == "2022") or (tenK[:4] == "2021") or (tenK[:4] == "2020"):
                file_path = os.path.join(directory3, tenK)
                with open(file_path, 'r') as file:
                    for line in file:
                        data = json.loads(line)
                        if 'paragraph' in data:
                            concatenated_string = " ".join(data["paragraph"])
                            id_to_paragraph[data['id']] = concatenated_string
                        
    
    with open('cik_to_ticker.json', 'r') as file:
        cik_to_ticker = json.load(file)

    keys_list = list(cik_to_ticker.keys())
    
    # Load the data
    with open('para_info_financial_encoder_best.pkl', 'rb') as f:
        para_info = pickle.load(f)

    years = ['2021', '2022', '2023']
    for year in years:
        filtered_para_info = [item for item in para_info if item[2].startswith(year)]
        filtered_para_info2 = [item for item in filtered_para_info if item[2].split('_')[2] in keys_list]
        filtered_para_info3 = [item for item in filtered_para_info2 if (item[2].split('_')[4] == "item1") or (item[2].split('_')[4] == "item1a") or (item[2].split('_')[4] == "item7a")]

        # Extract embeddings, texts, and ids from the filtered data
        final_embeddings = np.vstack([item[1] for item in filtered_para_info3]).astype('float32')
        final_texts = [item[0] for item in filtered_para_info3]
        final_ids = [item[2] for item in filtered_para_info3]

        # Create a FAISS index
        index = create_faiss_index(final_embeddings)

        intersection_graph = {outer_key: {inner_key: {} for inner_key in keys_list} for outer_key in keys_list}

        # Define batch size and other parameters
        batch_size = 512
        k = 5000
        threshold = 0.75

        # Iterate over the embeddings in batches
        for batch_start in range(0, len(final_embeddings), batch_size):
            print(batch_start)
            batch_end = min(batch_start + batch_size, len(final_embeddings))
            batch_embeddings = np.array(final_embeddings[batch_start:batch_end])
            batch_ids = final_ids[batch_start:batch_end]
            batch_texts = final_texts[batch_start:batch_end]

            # Perform batch search
            similar_indices, similar_scores = search_index(index, batch_embeddings, k)

            for i, (indices, scores) in enumerate(zip(similar_indices, similar_scores)):
                cik = batch_ids[i].split('_')[2]
                cap_paragraph = id_to_paragraph[batch_ids[i]]
                item = batch_ids[i].split('_')[4]
                cap_paragraph_with_item = f'{cap_paragraph} <strong>(From {item})</strong>'

                similar_ids = [final_ids[int(idx)] for idx, score in zip(indices, scores) if score > threshold]
                similar_texts = [final_texts[int(idx)] for idx, score in zip(indices, scores) if score > threshold]

                for similar_id, similar_text in zip(similar_ids, similar_texts):
                    if similar_id.split('_')[2] != cik:
                        cap_similar_paragraph = id_to_paragraph[similar_id]
                        similar_item = similar_id.split('_')[4]
                        cap_similar_paragraph_with_item = f'{cap_similar_paragraph} <strong>(From {similar_item})</strong>'
                        if (cap_paragraph_with_item in intersection_graph[cik][similar_id.split('_')[2]]) and (cap_similar_paragraph_with_item not in intersection_graph[cik][similar_id.split('_')[2]][cap_paragraph_with_item]):
                            intersection_graph[cik][similar_id.split('_')[2]][cap_paragraph_with_item].append(cap_similar_paragraph_with_item)
                        else:
                            intersection_graph[cik][similar_id.split('_')[2]][cap_paragraph_with_item] = [cap_similar_paragraph_with_item]

                        if (cap_similar_paragraph_with_item in intersection_graph[similar_id.split('_')[2]][cik]) and (cap_paragraph_with_item not in intersection_graph[similar_id.split('_')[2]][cik][cap_similar_paragraph_with_item]):
                            intersection_graph[similar_id.split('_')[2]][cik][cap_similar_paragraph_with_item].append(cap_paragraph_with_item)
                        else:
                            intersection_graph[similar_id.split('_')[2]][cik][cap_similar_paragraph_with_item] = [cap_paragraph_with_item]
        
        with open(f'firms_mutual_risk_paragraphs_{year}.json', 'w') as json_file:
            json.dump(intersection_graph, json_file, indent=4)

                        
if __name__ == "__main__":
    main()
