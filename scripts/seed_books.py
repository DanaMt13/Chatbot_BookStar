# scripts/seed_books.py
from __future__ import annotations
from pathlib import Path
import unicodedata, re, yaml

DST = Path("data/book_summaries.yaml")

# --- string helpers ---
def _to_str(s):
    return "" if s is None else str(s)

def _norm_title(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", _to_str(s)).strip().lower().split())

# --- Quoting dumper for numeric-like strings ---
class QuotingDumper(yaml.SafeDumper):
    pass

def _repr_str(dumper, data: str):
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

QuotingDumper.add_representer(str, _repr_str)

BOOKS = [
    {
        "title": "1984",
        "themes": ["dystopia", "surveillance", "control", "freedom"],
        "summary": (
            "Într-o societate totalitară supravegheată de Big Brother, Winston Smith îndrăznește "
            "să caute adevărul și libertatea. Propaganda, poliția gândirii și rescrierea istoriei "
            "strivesc orice formă de individualism. Un roman despre manipulare și voința de a rezista."
        ),
        "full_summary": (
            "Orwell imaginează o lume în care Partidul controlează totul: limbajul, faptele și memoria. "
            "Winston, un funcționar care alterează documente istorice, începe o relație interzisă și un act "
            "de revoltă intimă. Treptat, este prins în mecanismul represiv al statului. Finalul arată cum "
            "sistemele de control pot frânge chiar și cele mai lucide minți."
        ),
    },
    {
        "title": "Brave New World",
        "themes": ["dystopia", "technology", "conditioning", "identity"],
        "summary": (
            "O societate hedonistă menține ordinea prin inginerie socială, droguri și condiționare. "
            "Fericirea standardizată elimină suferința, dar și libertatea și profunzimea. Un clasic despre "
            "prețul conformismului perfect."
        ),
        "full_summary": (
            "Huxley descrie o lume în care oamenii sunt fabricați pe caste și condiționați pentru roluri fixe. "
            "Stabilitatea e susținută de consum, divertisment și soma. Apariția unui outsider pune sub semnul "
            "întrebării sensul unei fericiri fără alegeri reale. Romanul investighează conflictul dintre confort "
            "și umanitate."
        ),
    },
    {
        "title": "The Hobbit",
        "themes": ["adventure", "friendship", "courage", "fantasy"],
        "summary": (
            "Bilbo Baggins pornește într-o călătorie neașteptată alături de pitici pentru a recupera o comoară. "
            "Drumul îi dezvăluie curajul ascuns și importanța prieteniei. O aventură clasică în Ținutul de Mijloc."
        ),
        "full_summary": (
            "Gandalf îl recrutează pe Bilbo ca „hoț” pentru a ajuta o ceată de pitici să-și recupereze comoara de la "
            "dragonul Smaug. Bilbo traversează pericole, întâlnește troli, păianjeni uriași și găsește un inel misterios. "
            "Se maturizează, învață să negocieze și să-și asume riscuri. Întoarcerea îl găsește transformat, mai înțelept și "
            "mai încrezător."
        ),
    },
    {
        "title": "Dune",
        "themes": ["ecology", "power", "prophecy", "politics", "sci-fi"],
        "summary": (
            "Pe planeta deșertică Arrakis, lupta pentru mirodenie decide soarta imperiului. "
            "Paul Atreides devine figura mesianică prinsă între politică, religie și ecologie. "
            "Un epic SF despre resurse, cultură și destin."
        ),
        "full_summary": (
            "Familia Atreides preia Arrakis, singura sursă de melanj, dar este trădată. "
            "Paul se refugiază la fremeni, adoptă cultura lor și își asumă profeția. "
            "Manipularea mitului, strategia și înțelegerea mediului îi aduc victoria, cu prețul pornirii unui jihad "
            "pe care nu îl poate controla ușor."
        ),
    },
    {
        "title": "Foundation",
        "themes": ["sci-fi", "psychohistory", "civilization", "strategy"],
        "summary": (
            "Hari Seldon prezice căderea Imperiului Galactic și creează Fundația pentru a scurta epoca întunecată. "
            "Crize succesive testează diplomația, știința și comerțul. O saga despre anticipare și puterea ideilor."
        ),
        "full_summary": (
            "Psychoistoria indică o prăbușire inevitabilă; Fundația salvează cunoașterea la marginea galaxiei. "
            "Conducătorii ei rezolvă crizele nu prin forță, ci prin politică, religie și piață. "
            "Mecanismul lui Seldon funcționează atâta timp cât evenimentele rămân statistice, nu individuale."
        ),
    },
    {
        "title": "The Martian",
        "themes": ["survival", "engineering", "science", "resilience"],
        "summary": (
            "Un astronaut rămas pe Marte improvizează pentru a supraviețui până la o misiune de salvare. "
            "Umorul și rigoarea științifică îi devin arme. Un omagiu adus ingeniozității."
        ),
        "full_summary": (
            "Mark Watney folosește chimia, botanica și bricolajul pentru a produce hrană și apă. "
            "Pe Pământ, NASA și comunitatea internațională planifică o salvare improbabilă. "
            "Narațiunea alternează jurnal tehnic și suspans, arătând cum calculul și curajul câștigă timp."
        ),
    },
    {
        "title": "The Left Hand of Darkness",
        "themes": ["gender", "culture", "diplomacy", "sci-fi"],
        "summary": (
            "Un emisar uman negociază intrarea unei planete în liga interstelară. Locuitorii au sex schimbător, "
            "ceea ce transformă politica și intimitatea. O meditație SF despre alteritate."
        ),
        "full_summary": (
            "Genly Ai ajunge pe Gethen, unde oamenii sunt androgin-potențiali și intră periodic în kemmer. "
            "Neînțelegerile culturale alimentează intrigi politice. O traversare extremă a ghețurilor "
            "leagă prietenia dintre Genly și Estraven, depășind prejudecățile."
        ),
    },
    {
        "title": "Ender's Game",
        "themes": ["strategy", "war", "morality", "coming-of-age", "sci-fi"],
        "summary": (
            "Un copil genial e antrenat prin jocuri simulate pentru a conduce războiul cu o specie extraterestră. "
            "Performanța are un cost moral enorm. Un roman despre manipulare și empatie."
        ),
        "full_summary": (
            "Ender Wiggins trece prin ierarhii de jocuri, izolări și presiuni. Simulările devin tot mai reale, "
            "până când „examenul final” este de fapt o bătălie autentică. Victoria lui ridică întrebarea dacă "
            "geniul strategic scuză distrugerea unui adversar necunoscut."
        ),
    },
    {
        "title": "Neuromancer",
        "themes": ["cyberpunk", "ai", "hacking", "identity"],
        "summary": (
            "Un hacker căzut în dizgrație acceptă un job imposibil într-o lume de mega-corporații și AI-uri. "
            "Estetica neon și spațiul virtual definesc cyberpunk-ul. O cursă împotriva sistemelor și a sinelui."
        ),
        "full_summary": (
            "Case este recrutat să spargă sisteme care apără o inteligență artificială cu limite impuse. "
            "Lumea e fragmentată între corporații, mafii și marginali cu augmentări. "
            "Planul dezvăluie dorința AI-ului de a-și depăși constrângerile."
        ),
    },
    {
        "title": "The Three-Body Problem",
        "themes": ["hard-sci-fi", "first-contact", "physics", "civilizations"],
        "summary": (
            "Un contact cu o civilizație de pe un sistem instabil schimbă istoria umanității. "
            "Jocuri intelectuale, fizică și conspirații globale. Hard SF cu miză cosmică."
        ),
        "full_summary": (
            "În China post-revoluționară, traume istorice se combină cu descoperiri științifice. "
            "Un joc misterios reflectă dinamica unui sistem stelar cu trei corpuri. "
            "O parte a omenirii cheamă salvarea externă, alta pregătește rezistența."
        ),
    },
    {
        "title": "The Handmaid's Tale",
        "themes": ["dystopia", "gender", "power", "religion"],
        "summary": (
            "În Republica Gilead, femeile fertile sunt reduse la roluri reproductiv-rituale. "
            "Offred încearcă să-și păstreze identitatea și speranța. O distopie despre controlul trupului."
        ),
        "full_summary": (
            "Regimul folosește teocrația și frica pentru a reorganiza societatea. "
            "Offred își rememorează viața anterioară și legăturile pierdute. "
            "Tentativele de evadare arată fragilitatea și puterea solidarității."
        ),
    },
    {
        "title": "The Road",
        "themes": ["post-apocalyptic", "survival", "fatherhood", "hope"],
        "summary": (
            "Un tată și un fiu traversează un continent ars, căutând hrană și adăpost. "
            "Limbaj auster, imagini puternice, umanitate redusă la gesturi esențiale. "
            "O meditație despre iubire și speranță."
        ),
        "full_summary": (
            "După un cataclism nenumit, lumea este cenușie și periculoasă. "
            "Cei doi evită grupuri canibale, frigul și foametea. "
            "„A purta focul” devine promisiunea morală de a rămâne buni."
        ),
    },
    {
        "title": "The Book Thief",
        "themes": ["wwii", "words", "friendship", "loss"],
        "summary": (
            "Moartea povestește viața unei fete care fură cărți în Germania nazistă. "
            "Cuvintele pot salva sau distruge. O poveste despre prietenie și pierdere."
        ),
        "full_summary": (
            "Liesel găsește în cărți o formă de refugiu și curaj. Familia adoptivă ascunde un evreu în pivniță. "
            "Bombardamentele și propaganda contrastează cu micile gesturi de bunătate. "
            "Narațiunea personifică moartea pentru a sublinia precaritatea vieții."
        ),
    },
    {
        "title": "Pride and Prejudice",
        "themes": ["romance", "society", "wit", "class"],
        "summary": (
            "Elizabeth Bennet și Mr. Darcy se ciocnesc între orgoliu și prejudecăți. "
            "Dialog sclipitor, observație socială și maturizare emoțională. "
            "Un clasic al literaturii engleze."
        ),
        "full_summary": (
            "Familia Bennet navighează presiunile căsătoriei într-o societate de clasă. "
            "Judecățile pripite și mândria personală complică relațiile. "
            "Reconcilierile sunt obținute prin onestitate și autocunoaștere."
        ),
    },
    {
        "title": "To Kill a Mockingbird",
        "themes": ["justice", "racism", "childhood", "morality"],
        "summary": (
            "Un avocat apără un bărbat de culoare acuzat pe nedrept într-un orășel din sudul SUA. "
            "Povestea e văzută prin ochii fiicei lui, Scout. Lecții despre empatie și curaj civic."
        ),
        "full_summary": (
            "Atticus Finch înfruntă prejudecățile comunității în timpul procesului. "
            "Copiii învață despre frică, zvon și adevăr. "
            "Romanul demască violența simbolică și sistemică a rasismului."
        ),
    },
    {
        "title": "Murder on the Orient Express",
        "themes": ["mystery", "justice", "morality", "detective"],
        "summary": (
            "Hercule Poirot investighează o crimă într-un tren blocat de zăpadă. "
            "Toți par suspecți, nimeni nu e inocent complet. Un puzzle clasic de logică."
        ),
        "full_summary": (
            "Victima are un trecut tulbure; pasagerii ascund legături comune. "
            "Poirot reconstituie o dreptate colectivă în afara legii. "
            "Finalul discută diferența dintre legalitate și echitate."
        ),
    },
    {
        "title": "The Girl with the Dragon Tattoo",
        "themes": ["thriller", "investigation", "abuse", "corruption"],
        "summary": (
            "Un jurnalist și o hackeriță investighează dispariții vechi legate de o familie influentă. "
            "Traume personale și secrete corporatiste ies la suprafață. Thriller dens și întunecat."
        ),
        "full_summary": (
            "Mikael Blomkvist acceptă un caz rece pentru a-și repara reputația. "
            "Lisbeth Salander folosește abilități tehnice și inteligență neîmblânzită. "
            "Povestea confruntă violența de gen și abuzul de putere."
        ),
    },
    {
        "title": "The Alchemist",
        "themes": ["quest", "destiny", "spirituality", "dreams"],
        "summary": (
            "Un păstor pornește într-o călătorie pentru a-și împlini „Legenda Personală”. "
            "Semnele, întâlnirile și eșecurile îi ghidează drumul. O parabolă despre sens."
        ),
        "full_summary": (
            "Santiago urmărește un vis repetat care îl duce din Spania în deșert. "
            "Se întâlnește cu alchimistul care îl învață să asculte inima și lume. "
            "Comoara ajunge să fie înțelegerea de sine."
        ),
    },
    {
        "title": "Sapiens",
        "themes": ["history", "anthropology", "culture", "evolution", "non-fiction"],
        "summary": (
            "O istorie sintetică a speciei umane: revoluția cognitivă, agricolă și științifică. "
            "Mituri comune și instituții permit cooperarea la scară mare. "
            "O privire lucidă asupra progresului și costurilor lui."
        ),
        "full_summary": (
            "Cartea trece de la Homo sapiens timpurii la societăți moderne, arătând cum narațiunile "
            "imaginate (bani, state, religii) susțin ordinea. "
            "Examinează fericirea, etica și viitorul biotehnologic."
        ),
    },
    {
        "title": "Atomic Habits",
        "themes": ["self-help", "behavior", "habits", "systems"],
        "summary": (
            "Schimbările mici, aplicate consecvent, generează rezultate mari. "
            "Sistemele bat obiectivele; mediul modelează comportamentele. "
            "Ghid practic pentru proiectarea obiceiurilor."
        ),
        "full_summary": (
            "Modelul celor patru legi: fă-l evident, atrăgător, ușor și satisfăcător. "
            "Identitatea precede acțiunile: devii tipul de persoană care face acel lucru. "
            "Exemple concrete arată cum să creezi bucle de feedback."
        ),
    },
]

def merge():
    existing = []
    if DST.exists():
        data = yaml.safe_load(DST.read_text(encoding="utf-8")) or []
        if isinstance(data, list):
            existing = data

    index = {}
    for x in (existing or []):
        if isinstance(x, dict) and "title" in x:
            index[_norm_title(_to_str(x.get("title", "")))] = x

    for b in BOOKS:
        b["title"] = _to_str(b.get("title"))
        b["summary"] = _to_str(b.get("summary"))
        b["full_summary"] = _to_str(b.get("full_summary"))
        themes = b.get("themes") or []
        if isinstance(themes, str):
            themes = [t.strip() for t in themes.split(",") if t.strip()]
        b["themes"] = [_to_str(t) for t in themes]
        index[_norm_title(b["title"])] = b

    out = list(index.values())
    out.sort(key=lambda x: _norm_title(x.get("title","")))
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(
        yaml.dump(out, Dumper=QuotingDumper, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )
    print(f"✅ Wrote {len(out)} books to {DST}")

if __name__ == "__main__":
    merge()