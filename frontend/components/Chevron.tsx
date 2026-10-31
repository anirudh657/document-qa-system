/** Disclosure arrow. Decorative: the `<summary>` text is what gets announced. */
export default function Chevron({ size = 13 }: { size?: number }) {
  return (
    <svg
      className="chevron"
      width={size}
      height={size}
      viewBox="0 0 14 14"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M4 2.5L8.5 7 4 11.5"
        stroke="var(--faint)"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
