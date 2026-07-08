export const meta = {
  name: 'venue-trend-report',
  description: 'Summarize per-subfield research trends for a conference\'s accepted LLM-inference papers and produce a Markdown report',
  phases: [
    { title: 'Trend analysis' },
  ],
}

const TREND_SCHEMA = {
  type: 'object',
  properties: {
    core_problems: { type: 'string' },
    representative_papers: { type: 'array', items: { type: 'string' } },
    method_commonalities: { type: 'string' },
  },
  required: ['core_problems', 'representative_papers', 'method_commonalities'],
}

// args.groups: [{ subfield: string, papers: [{title, abstract, url}] }]
const groups = args.groups

const summaries = await pipeline(groups, (group) => {
  const paperList = group.papers
    .map((p) => `- ${p.title}\n  ${p.abstract}\n  (${p.url})`)
    .join('\n')
  return agent(
    `You are analyzing this conference's accepted papers in the "${group.subfield}" ` +
    `subfield of LLM inference deployment optimization. Papers:\n\n${paperList}\n\n` +
    `Summarize: (1) the core problems this subfield's papers are tackling, ` +
    `(2) 2-4 representative paper titles, (3) commonalities/divergences across their methods.`,
    { label: `trend:${group.subfield}`, phase: 'Trend analysis', schema: TREND_SCHEMA }
  ).then((trend) => ({ subfield: group.subfield, papers: group.papers, trend }))
})

const results = summaries.filter(Boolean).sort((a, b) => b.papers.length - a.papers.length)
log(`Summarized ${results.length}/${groups.length} subfields`)

const distributionTable = results
  .map((r) => `| ${r.subfield} | ${r.papers.length} |`)
  .join('\n')

const sections = results
  .map((r) => {
    const paperLinks = r.papers.map((p) => `- [${p.title}](${p.url})`).join('\n')
    const coreProblems = r.trend?.core_problems ?? '(analysis failed)'
    const reps = r.trend?.representative_papers?.map((t) => `- ${t}`).join('\n') ?? '(analysis failed)'
    const commonalities = r.trend?.method_commonalities ?? '(analysis failed)'
    return (
      `## ${r.subfield} (${r.papers.length} papers)\n\n` +
      `**Core problems:** ${coreProblems}\n\n` +
      `**Representative papers:**\n${reps}\n\n` +
      `**Method commonalities:** ${commonalities}\n\n` +
      `**All papers in this subfield:**\n${paperLinks}\n`
    )
  })
  .join('\n')

const report =
  `# ${args.title ?? 'Conference'} — LLM Inference Deployment Optimization Trend Report\n\n` +
  `## Subfield distribution\n\n| Subfield | # Papers |\n|---|---|\n${distributionTable}\n\n` +
  sections

return { report }
