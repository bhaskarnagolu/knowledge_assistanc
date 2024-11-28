export function titleCaps(e: string) {
    if (!e) return;
    return e
      .split('-')
      .map((el: string) => {
        return `${el.substring(0, 1).toUpperCase()}${el
          .substring(1, el.length)
          .toLowerCase()}`;
      })
      .join(' ');
  }
  