import MatchExplorerClient from "@/components/MatchExplorerClient";
import { MATCH_IDS, getExplorerData } from "@/lib/data";

export default function ExplorerPage() {
  const matches = MATCH_IDS.map((id) => {
    const { label, rows } = getExplorerData(id);
    return { matchId: id, label, rows };
  });
  return <MatchExplorerClient matches={matches} />;
}
