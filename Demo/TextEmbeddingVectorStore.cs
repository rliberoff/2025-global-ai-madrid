#pragma warning disable SKEXP0001

using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Embeddings;

namespace Demo;

public class TextEmbeddingVectorStore : IVectorStore
{
    private readonly IVectorStore decoratedVectorStore;

    private readonly ITextEmbeddingGenerationService textEmbeddingGenerationService;

    public TextEmbeddingVectorStore(IVectorStore decoratedVectorStore, ITextEmbeddingGenerationService textEmbeddingGenerationService)
    {
        this.decoratedVectorStore = decoratedVectorStore ?? throw new ArgumentNullException(nameof(decoratedVectorStore));
        this.textEmbeddingGenerationService = textEmbeddingGenerationService ?? throw new ArgumentNullException(nameof(textEmbeddingGenerationService));
    }

    public IVectorStoreRecordCollection<TKey, TRecord> GetCollection<TKey, TRecord>(string name, VectorStoreRecordDefinition? vectorStoreRecordDefinition = null)
        where TKey : notnull
    {
        var collection = this.decoratedVectorStore.GetCollection<TKey, TRecord>(name, vectorStoreRecordDefinition);
        var embeddingStore = new TextEmbeddingVectorStoreRecordCollection<TKey, TRecord>(collection, this.textEmbeddingGenerationService);
        return embeddingStore;
    }

    public IAsyncEnumerable<string> ListCollectionNamesAsync(CancellationToken cancellationToken = default)
    {
        return this.decoratedVectorStore.ListCollectionNamesAsync(cancellationToken);
    }
}

