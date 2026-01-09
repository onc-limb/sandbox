"use client";

const user = {
  name: "Hedy Lamarr",
  imageUrl: "https://i.imgur.com/yXOvdOSs.jpg",
  imageSize: 90,
};

const isLoggedIn = true;

function MyButton() {
  function handleClick() {
    alert("You clicked me!");
  }
  return <button onClick={handleClick}>Click me</button>;
}

export default function Home() {
  return (
    <div>
      <h1>welcomw to my app</h1>
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
