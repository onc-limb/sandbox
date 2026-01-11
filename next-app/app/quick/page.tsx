"use client";

import { useState } from "react";

const user = {
  name: "Hedy Lamarr",
  imageUrl: "https://i.imgur.com/yXOvdOSs.jpg",
  imageSize: 90,
};

const isLoggedIn = true;

function MyButton() {
  const [count, setCount] = useState(0);
  function handleClick() {
    setCount(count + 1);
  }
  return <button onClick={handleClick}>Click {count} times</button>;
}

export default function Home() {
  return (
    <div>
      <h1>welcomw to my app</h1>
      {isLoggedIn && <MyButton />}
      {isLoggedIn && <MyButton />}
      <img
        className="avatar"
        src={user.imageUrl}
        alt={"Photo of " + user.name}
        style={{
          width: user.imageSize,
          height: user.imageSize,
        }}
      />
    </div>
  );
}
