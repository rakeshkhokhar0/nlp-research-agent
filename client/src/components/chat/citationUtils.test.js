import test from 'node:test';
import assert from 'node:assert/strict';
import { getUniqueCitations, formatCitationText } from './citationUtils.js';

test('citationUtils - deduplicates citations with identical paper title and page number', () => {
  const rawCitations = [
    {
      chunk_id: 'chunk-1',
      paper_title: 'Attention Is All You Need',
      page_number: 3,
      relevance_score: 4.95,
      quote_text: 'The Transformer uses multi-head attention.',
    },
    {
      chunk_id: 'chunk-2',
      paper_title: 'Attention Is All You Need',
      page_number: 3,
      relevance_score: 4.82,
      quote_text: 'Self-attention layers connect all positions.',
    },
    {
      chunk_id: 'chunk-3',
      paper_title: 'Attention Is All You Need',
      page_number: 5,
      relevance_score: 4.10,
      quote_text: 'Positional encodings add order information.',
    },
  ];

  const result = getUniqueCitations(rawCitations);

  assert.equal(result.length, 2);
  assert.equal(result[0].paper_title, 'Attention Is All You Need');
  assert.equal(result[0].page_number, 3);
  assert.equal(result[1].paper_title, 'Attention Is All You Need');
  assert.equal(result[1].page_number, 5);

  // Ensure internal retrieval metadata is NOT present in sanitized citation objects
  assert.equal(result[0].chunk_id, undefined);
  assert.equal(result[0].relevance_score, undefined);
  assert.equal(result[0].quote_text, undefined);
});

test('citationUtils - formats reference display string with title and page only', () => {
  const citationWithPage = {
    paper_title: 'Attention Is All You Need',
    page_number: 3,
  };
  assert.equal(formatCitationText(citationWithPage), 'Attention Is All You Need — Page 3');

  const citationWithoutPage = {
    paper_title: 'BERT: Pre-training of Deep Bidirectional Transformers',
    page_number: null,
  };
  assert.equal(formatCitationText(citationWithoutPage), 'BERT: Pre-training of Deep Bidirectional Transformers');
});

test('citationUtils - handles null or empty input gracefully', () => {
  assert.deepEqual(getUniqueCitations(null), []);
  assert.deepEqual(getUniqueCitations([]), []);
  assert.deepEqual(getUniqueCitations([{ chunk_id: '1' }]), []);
  assert.equal(formatCitationText(null), '');
  assert.equal(formatCitationText({}), '');
});
