import MatchAnalyserClient from "@/components/MatchAnalyserClient";
import { getAnalyserData } from "@/lib/data";

export default function AnalyserPage() {
  const matches = getAnalyserData();
  return <MatchAnalyserClient matches={matches} />;
}
