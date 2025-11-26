import torch
import torch.nn as nn
import esm
from pathlib import Path
from typing import Dict, List, Optional, Union
from Bio import SeqIO
import pandas as pd
from collections import Counter

class DataCleaner:
    def __init__(self, min_length: int = 50, max_length: int = 1024):
        self.min_length = min_length
        self.max_length = max_length
        self.standard_aas = set('ACDEFGHIKLMNPQRSTVWY')
        
    def clean_sequence(self, sequence: str) -> Optional[str]:
        """Clean a single protein sequence"""
        # Remove whitespace and convert to uppercase
        seq = ''.join(sequence.split()).upper()
        
        # Filter by length
        if len(seq) < self.min_length:
            return None
            
        # Truncate if too long (for ESM model constraints)
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]
            
        # Check if sequence contains only standard AAs
        if not all(aa in self.standard_aas for aa in seq):
            # Option: remove or handle non-standard AAs
            # For now, we'll keep them as ESM can handle some non-standard
            pass
            
        return seq
    
    def load_and_clean_fasta(self, fasta_path: Union[str, Path]) -> Dict[str, str]:
        """Load and clean FASTA file"""
        path = Path(fasta_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"FASTA file not found at {path}")

        clean_sequences = {}
        
        for record in SeqIO.parse(str(path), "fasta"):
            clean_seq = self.clean_sequence(str(record.seq))
            if clean_seq:
                clean_sequences[record.id] = clean_seq
                
        return clean_sequences
    
    def load_go_terms(self, terms_path: str) -> Dict[str, List[str]]:
        """Load GO terms and group by protein"""
        df = pd.read_csv(terms_path, sep='\t')
        go_terms = {}
        
        for _, row in df.iterrows():
            protein_id = row['EntryID']
            term = row['term']
            
            if protein_id not in go_terms:
                go_terms[protein_id] = []
            go_terms[protein_id].append(term)
            
        return go_terms
    
    def load_taxonomy(self, taxonomy_path: Union[str, Path]) -> Dict[str, int]:
        """Load taxonomy mapping"""
        path = Path(taxonomy_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Taxonomy file not found at {path}")

        # The provided taxonomy file has no header, so we coerce one and drop invalid rows.
        df = pd.read_csv(
            path,
            sep='\t',
            header=None,
            names=['EntryID', 'TaxID'],
            comment='#',
        )
        df['TaxID'] = pd.to_numeric(df['TaxID'], errors='coerce')
        df = df.dropna(subset=['TaxID'])
        df['TaxID'] = df['TaxID'].astype(int)
        return dict(zip(df['EntryID'], df['TaxID']))

class ESMEmbedder:
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D"):
        self.model_name = model_name
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Set model to eval mode
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """Embed a single sequence"""
        batch_labels, batch_strs, batch_tokens = self.batch_converter([("protein", sequence)])
        
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()
            
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])
            embeddings = results["representations"][33]
            
        # Remove CLS and EOS tokens, take mean over sequence
        sequence_embedding = embeddings[0, 1:-1].mean(dim=0)
        return sequence_embedding.cpu()
    
    def embed_batch(self, sequences: List[str], batch_size: int = 8) -> torch.Tensor:
        """Embed a batch of sequences"""
        all_embeddings = []
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch_data = [(f"protein_{i}", seq) for i, seq in enumerate(batch_seqs)]
            
            batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
            
            if torch.cuda.is_available():
                batch_tokens = batch_tokens.cuda()
                
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33])
                embeddings = results["representations"][33]
                
            # Mean pool over sequence length (excluding CLS and EOS)
            batch_embeddings = embeddings[:, 1:-1, :].mean(dim=1)
            all_embeddings.append(batch_embeddings.cpu())
            
        return torch.cat(all_embeddings, dim=0)

class KmerEmbedder:
    def __init__(self, k: int = 3):
        self.k = k
        self.vocab = {}
        self.vocab_size = 0
        self.embedding_dim = 128  # You can adjust this
        
    def build_vocab(self, sequences: List[str], min_freq: int = 1):
        """Build k-mer vocabulary from sequences"""
        kmer_counter = Counter()
        
        for seq in sequences:
            kmers = self._get_kmers(seq)
            kmer_counter.update(kmers)
            
        # Create vocabulary, filtering by frequency
        self.vocab = {}
        idx = 0
        for kmer, count in kmer_counter.items():
            if count >= min_freq:
                self.vocab[kmer] = idx
                idx += 1
                
        # Add unknown token
        self.vocab['<UNK>'] = idx
        self.vocab_size = len(self.vocab)
        
        # Initialize embedding layer
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        
    def _get_kmers(self, sequence: str) -> List[str]:
        """Extract k-mers from sequence"""
        return [sequence[i:i+self.k] for i in range(len(sequence) - self.k + 1)]
    
    def sequence_to_kmers(self, sequence: str) -> List[int]:
        """Convert sequence to k-mer indices"""
        kmers = self._get_kmers(sequence)
        return [self.vocab.get(kmer, self.vocab['<UNK>']) for kmer in kmers]
    
    def embed_sequence(self, sequence: str, pooling: str = 'mean') -> torch.Tensor:
        """Embed a single sequence using k-mers"""
        kmer_indices = self.sequence_to_kmers(sequence)
        
        if not kmer_indices:
            return torch.zeros(self.embedding_dim)
            
        kmer_tensor = torch.tensor(kmer_indices)
        embeddings = self.embedding_layer(kmer_tensor)
        
        if pooling == 'mean':
            return embeddings.mean(dim=0)
        elif pooling == 'max':
            return embeddings.max(dim=0)[0]
        else:
            return embeddings.mean(dim=0)

class TaxonomicEmbedder:
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.taxid_to_idx = {}
        self.unknown_index = 0
        self.embedding_layer = None
        
    def build_vocab(self, taxon_ids: List[int]):
        """Build taxonomy vocabulary"""
        unique_taxids = sorted({taxid for taxid in taxon_ids if taxid is not None})
        self.taxid_to_idx = {taxid: idx + 1 for idx, taxid in enumerate(unique_taxids)}
        self.vocab_size = len(unique_taxids) + 1  # Reserve index 0 for unknown
        
        # Initialize embedding layer
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        
    def embed_taxid(self, taxid: Optional[int]) -> torch.Tensor:
        """Embed a single taxon ID"""
        if self.embedding_layer is None:
            raise ValueError("Taxonomic vocabulary has not been built. Call build_vocab first.")

        if taxid is None:
            idx = self.unknown_index
        else:
            idx = self.taxid_to_idx.get(taxid, self.unknown_index)
        return self.embedding_layer(torch.tensor(idx, dtype=torch.long))

class MultiModalEmbedder:
    def __init__(self, esm_embedder: ESMEmbedder, 
                 kmer_embedder: Optional[KmerEmbedder] = None,
                 tax_embedder: Optional[TaxonomicEmbedder] = None):
        self.esm_embedder = esm_embedder
        self.kmer_embedder = kmer_embedder
        self.tax_embedder = tax_embedder
        
    def embed_protein(self, sequence: str, taxid: Optional[int] = None) -> torch.Tensor:
        """Create multi-modal embedding for a protein"""
        embeddings = []
        
        # ESM embeddings (main)
        esm_emb = self.esm_embedder.embed_sequence(sequence)
        embeddings.append(esm_emb)
        
        # K-mer embeddings (optional)
        if self.kmer_embedder:
            kmer_emb = self.kmer_embedder.embed_sequence(sequence)
            embeddings.append(kmer_emb)
            
        # Taxonomic embeddings (optional)
        if self.tax_embedder:
            tax_emb = self.tax_embedder.embed_taxid(taxid)
            embeddings.append(tax_emb)
            
        # Concatenate all embeddings
        return torch.cat(embeddings, dim=0)

# Usage Example:
def main():
    # Initialize components
    cleaner = DataCleaner(min_length=50, max_length=1024)
    esm_embedder = ESMEmbedder()
    
    # Load and clean data
    data_dir = Path(__file__).resolve().parent / "data" / "Train"
    sequences_path = data_dir / "train_sequences.fasta"
    taxonomy_path = data_dir / "train_taxonomy.tsv"

    sequences = cleaner.load_and_clean_fasta(sequences_path)
    if not sequences:
        raise ValueError(f"No valid sequences were loaded from {sequences_path}.")

    taxonomy = cleaner.load_taxonomy(taxonomy_path)
    
    # Build k-mer vocabulary
    kmer_embedder = KmerEmbedder(k=3)
    kmer_embedder.build_vocab(list(sequences.values()), min_freq=2)
    
    # Build taxonomic vocabulary
    tax_embedder = TaxonomicEmbedder(embedding_dim=64)
    tax_embedder.build_vocab(list(taxonomy.values()))
    
    # Create multi-modal embedder
    multi_embedder = MultiModalEmbedder(
        esm_embedder=esm_embedder,
        kmer_embedder=kmer_embedder,
        tax_embedder=tax_embedder
    )
    
    # Example: Embed a protein
    protein_id = next(iter(sequences))
    sequence = sequences[protein_id]
    taxid = taxonomy.get(protein_id)
    
    embedding = multi_embedder.embed_protein(sequence, taxid)
    print(f"Final embedding dimension: {embedding.shape}")

if __name__ == "__main__":
    main()