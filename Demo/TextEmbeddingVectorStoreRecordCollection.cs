#pragma warning disable SKEXP0001

using System.Reflection;
using System.Runtime.CompilerServices;

using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Embeddings;

namespace Demo;

public class TextEmbeddingVectorStoreRecordCollection<TKey, TRecord> : IVectorStoreRecordCollection<TKey, TRecord>, IVectorizableTextSearch<TRecord>
    where TKey : notnull
{
    private readonly IVectorStoreRecordCollection<TKey, TRecord> decoratedVectorStoreRecordCollection;

    private readonly ITextEmbeddingGenerationService textEmbeddingGenerationService;

    private readonly IEnumerable<(PropertyInfo EmbeddingPropertyInfo, IList<PropertyInfo> SourcePropertiesInfo)> embeddingPropertiesWithSourceProperties;

    public TextEmbeddingVectorStoreRecordCollection(IVectorStoreRecordCollection<TKey, TRecord> decoratedVectorStoreRecordCollection, ITextEmbeddingGenerationService textEmbeddingGenerationService)
    {
        // Assign.
        this.decoratedVectorStoreRecordCollection = decoratedVectorStoreRecordCollection ?? throw new ArgumentNullException(nameof(decoratedVectorStoreRecordCollection));
        this.textEmbeddingGenerationService = textEmbeddingGenerationService ?? throw new ArgumentNullException(nameof(textEmbeddingGenerationService));

        // Find all the embedding properties to generate embeddings for.
        this.embeddingPropertiesWithSourceProperties = FindDataPropertiesWithEmbeddingProperties(typeof(TRecord));
    }

    /// <inheritdoc />
    public string CollectionName => this.decoratedVectorStoreRecordCollection.CollectionName;

    /// <inheritdoc />
    public Task<bool> CollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        return this.decoratedVectorStoreRecordCollection.CollectionExistsAsync(cancellationToken);
    }

    /// <inheritdoc />
    public Task CreateCollectionAsync(CancellationToken cancellationToken = default)
    {
        return this.decoratedVectorStoreRecordCollection.CreateCollectionAsync(cancellationToken);
    }

    /// <inheritdoc />
    public async Task CreateCollectionIfNotExistsAsync(CancellationToken cancellationToken = default)
    {
        if (!await this.CollectionExistsAsync(cancellationToken).ConfigureAwait(false))
        {
            await this.CreateCollectionAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    /// <inheritdoc />
    public Task DeleteCollectionAsync(CancellationToken cancellationToken = default)
    {
        return this.decoratedVectorStoreRecordCollection.DeleteCollectionAsync(cancellationToken);
    }

    /// <inheritdoc />
    public Task DeleteAsync(TKey key, DeleteRecordOptions? options = null, CancellationToken cancellationToken = default)
    {
        return this.decoratedVectorStoreRecordCollection.DeleteAsync(key, options, cancellationToken);
    }

    /// <inheritdoc />
    public Task DeleteBatchAsync(IEnumerable<TKey> keys, DeleteRecordOptions? options = null, CancellationToken cancellationToken = default)
    {
        return this.decoratedVectorStoreRecordCollection.DeleteBatchAsync(keys, options, cancellationToken);
    }

    /// <inheritdoc />
    public Task<TRecord?> GetAsync(TKey key, GetRecordOptions? options = null, CancellationToken cancellationToken = default)
    {
        return this.decoratedVectorStoreRecordCollection.GetAsync(key, options, cancellationToken);
    }

    /// <inheritdoc />
    public IAsyncEnumerable<TRecord> GetBatchAsync(IEnumerable<TKey> keys, GetRecordOptions? options = null, CancellationToken cancellationToken = default)
    {
        return this.decoratedVectorStoreRecordCollection.GetBatchAsync(keys, options, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<TKey> UpsertAsync(TRecord record, UpsertRecordOptions? options = null, CancellationToken cancellationToken = default)
    {
        var recordWithEmbeddings = await this.AddEmbeddingsAsync(record, cancellationToken).ConfigureAwait(false);
        return await this.decoratedVectorStoreRecordCollection.UpsertAsync(recordWithEmbeddings, options, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<TKey> UpsertBatchAsync(IEnumerable<TRecord> records, UpsertRecordOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var recordWithEmbeddingsTasks = records.Select(r => this.AddEmbeddingsAsync(r, cancellationToken));
        var recordWithEmbeddings = await Task.WhenAll(recordWithEmbeddingsTasks).ConfigureAwait(false);
        var upsertResults = this.decoratedVectorStoreRecordCollection.UpsertBatchAsync(recordWithEmbeddings, options, cancellationToken);
        await foreach (var upsertResult in upsertResults.ConfigureAwait(false))
        {
            yield return upsertResult;
        }
    }

    /// <inheritdoc />
    public Task<VectorSearchResults<TRecord>> VectorizedSearchAsync<TVector>(TVector vector, VectorSearchOptions? options = null, CancellationToken cancellationToken = default)
    {
        return this.decoratedVectorStoreRecordCollection.VectorizedSearchAsync(vector, options, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<VectorSearchResults<TRecord>> VectorizableTextSearchAsync(string searchText, VectorSearchOptions? options = null, CancellationToken cancellationToken = default)
    {
        var embeddingValue = await this.textEmbeddingGenerationService.GenerateEmbeddingAsync(searchText, cancellationToken: cancellationToken).ConfigureAwait(false);
        return await this.VectorizedSearchAsync(embeddingValue, options, cancellationToken).ConfigureAwait(false);
    }

    private async Task<TRecord> AddEmbeddingsAsync(TRecord record, CancellationToken cancellationToken)
    {
        foreach (var (embeddingPropertyInfo, sourcePropertiesInfo) in this.embeddingPropertiesWithSourceProperties)
        {
            var sourceValues = sourcePropertiesInfo.Select(x => x.GetValue(record)).Cast<string>().Where(x => !string.IsNullOrWhiteSpace(x));
            var sourceString = string.Join("\n", sourceValues);

            var embeddingValue = await this.textEmbeddingGenerationService.GenerateEmbeddingAsync(sourceString, cancellationToken: cancellationToken).ConfigureAwait(false);
            embeddingPropertyInfo.SetValue(record, embeddingValue);
        }

        return record;
    }

    private static IEnumerable<(PropertyInfo EmbeddingPropertyInfo, IList<PropertyInfo> SourcePropertiesInfo)> FindDataPropertiesWithEmbeddingProperties(Type dataModelType)
    {
        var allProperties = dataModelType.GetProperties();
        var propertiesDictionary = allProperties.ToDictionary(p => p.Name);

        // Loop through all the properties to find the ones that have the GenerateTextEmbeddingAttribute.
        foreach (var property in allProperties)
        {
            var attribute = property.GetCustomAttribute<GenerateTextEmbeddingAttribute>();
            if (attribute is not null)
            {
                // Find the source properties that the embedding should be generated from.
                var sourcePropertiesInfo = new List<PropertyInfo>();
                foreach (var sourcePropertyName in attribute.SourcePropertyNames)
                {
                    if (!propertiesDictionary.TryGetValue(sourcePropertyName, out var sourcePropertyInfo))
                    {
                        throw new ArgumentException($"The source property '{sourcePropertyName}' as referenced by embedding property '{property.Name}' does not exist in the record model.");
                    }
                    else if (sourcePropertyInfo.PropertyType != typeof(string))
                    {
                        throw new ArgumentException($"The source property '{sourcePropertyName}' as referenced by embedding property '{property.Name}' has type {sourcePropertyInfo.PropertyType} but must be a string.");
                    }
                    else
                    {
                        sourcePropertiesInfo.Add(sourcePropertyInfo);
                    }
                }

                yield return (property, sourcePropertiesInfo);
            }
        }
    }
}

