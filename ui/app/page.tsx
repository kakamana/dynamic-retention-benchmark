"use client";

import { useState } from "react";
import {
  QueryClient,
  QueryClientProvider,
  useMutation,
} from "@tanstack/react-query";

const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";
const SECTORS = [
  "Health", "ICT", "Finance", "Public", "Construction",
  "Hospitality", "Education", "Retail",
];

type Action = { action: string; mean_reward: number; n: number };
type BenchmarkResponse = {
  sector: string;
  sector_median: number;
  gap: number;
  percentile_rank: number;
  predicted_retention: number | null;
  top_actions: Action[];
  disclaimer: string;
};

const qc = new QueryClient();

export default function Page() {
  return (
    <QueryClientProvider client={qc}>
      <Home />
    </QueryClientProvider>
  );
}

function Home() {
  const [form, setForm] = useState({
    sector: "ICT",
    headcount: 800,
    comp_percentile: 55,
    training_hours: 32,
    manager_quality: 3.6,
    retention_rate: 0.74,
  });

  const mutation = useMutation<BenchmarkResponse, Error>({
    mutationFn: async () => {
      const r = await fetch(`${API}/benchmark`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!r.ok) throw new Error("Failed");
      return r.json();
    },
  });

  const data = mutation.data;

  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">Dynamic Retention Benchmarking</h1>
      <p className="opacity-70 mb-6">
        Where you stand vs your sector — and which lever has historically returned the most retention.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
        <Field label="Sector">
          <select
            value={form.sector}
            onChange={(e) => setForm({ ...form, sector: e.target.value })}
            className="w-full rounded-xl border px-3 py-2"
          >
            {SECTORS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </Field>
        <NumberField label="Headcount" value={form.headcount} onChange={(v) => setForm({ ...form, headcount: v })} />
        <NumberField label="Comp percentile" value={form.comp_percentile} onChange={(v) => setForm({ ...form, comp_percentile: v })} />
        <NumberField label="Training hours" value={form.training_hours} onChange={(v) => setForm({ ...form, training_hours: v })} />
        <NumberField label="Manager quality (1-5)" step={0.1} value={form.manager_quality} onChange={(v) => setForm({ ...form, manager_quality: v })} />
        <NumberField label="Retention rate" step={0.01} value={form.retention_rate} onChange={(v) => setForm({ ...form, retention_rate: v })} />
      </div>

      <button
        onClick={() => mutation.mutate()}
        disabled={mutation.isPending}
        className="rounded-xl px-4 py-2 bg-black text-white disabled:opacity-50"
      >
        {mutation.isPending ? "Benchmarking…" : "Benchmark this org"}
      </button>

      {data && (
        <>
          <div className="mt-8 grid grid-cols-3 gap-4">
            <Stat label="Sector median" value={(data.sector_median * 100).toFixed(1) + "%"} />
            <Stat label="Gap (you − sector)" value={(data.gap * 100).toFixed(1) + " pts"} />
            <Stat label="Percentile rank" value={(data.percentile_rank * 100).toFixed(0) + "%"} />
          </div>

          <div className="mt-6 rounded-2xl border p-4">
            <div className="text-xs uppercase opacity-60">Top recommended actions</div>
            <ol className="mt-2 space-y-1">
              {data.top_actions.map((a, i) => (
                <li key={i} className="flex justify-between">
                  <span>{i + 1}. {a.action.replace(/_/g, " ")}</span>
                  <span className="opacity-70 text-sm">
                    +{(a.mean_reward * 100).toFixed(2)} pts (n = {a.n})
                  </span>
                </li>
              ))}
            </ol>
          </div>
          <p className="mt-6 text-xs opacity-60 italic">{data.disclaimer}</p>
        </>
      )}
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border p-4">
      <div className="text-xs uppercase tracking-wide opacity-60">{label}</div>
      <div className="text-2xl font-semibold mt-1">{value}</div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <div className="text-xs uppercase opacity-60 mb-1">{label}</div>
      {children}
    </label>
  );
}

function NumberField({
  label, value, onChange, step = 1,
}: { label: string; value: number; onChange: (v: number) => void; step?: number }) {
  return (
    <Field label={label}>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full rounded-xl border px-3 py-2"
      />
    </Field>
  );
}
