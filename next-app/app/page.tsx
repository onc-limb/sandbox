import Link from "next/link";

export default function Page() {
  return (
    <div>
      <h1>React Tutorial</h1>
      <ul>
        <li>
          <Link href="/quick">Quicl</Link>
        </li>
        <li>
          <Link href="/tic-tac-toe">Tic-Tac-Toe</Link>
        </li>
      </ul>
    </div>
  );
}
